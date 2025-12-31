from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs.base.him_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .big_reddog_him_config import BigReddogHimRoughCfgJack
COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET

class BigReddogHimRoughJack(LeggedRobot):
    cfg : BigReddogHimRoughCfgJack
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.actuator_net_input = torch.zeros(self.num_envs*self.num_actions, 6, device=self.device, requires_grad=False)
        self.joint_pos_err_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_vel_err_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_pos_err_last_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.joint_vel_err_last_last = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.joint_pos_err_last[env_ids] = 0.
        self.joint_pos_err_last_last[env_ids] = 0.
        self.joint_vel_err_last[env_ids] = 0.
        self.joint_vel_err_last_last[env_ids] = 0.
    
    def get_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.root_states[:, 2:3]
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
    
    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
        # print("heights shape:", heights.shape)      # -> [num_envs, N]
        # print("heights[0] shape:", heights[0].shape) # -> [N]
        # print("heights[0]:", heights[0])           # -> 具體數值
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)
        
    def get_current_obs(self):
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)

        return current_obs
            
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.33 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            print("height",heights[0,:44])
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        # print("height",heights[0,:40])
        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions
    
    def _compute_torques(self, actions):
        if self.cfg.control.use_actuator_network:
            action_scaled = actions * self.cfg.control.action_scale
            joint_pos_des = self.default_dof_pos + action_scaled
            joint_pos_err = self.dof_pos - joint_pos_des    # shape: (num_envs, num_actions)
            joint_vel_err = self.dof_vel    # shape: (num_envs, num_actions)
            with torch.inference_mode():
                self.actuator_net_input = torch.cat((
                    joint_pos_err.unsqueeze(-1),
                    self.joint_pos_err_last.unsqueeze(-1),
                    self.joint_pos_err_last_last.unsqueeze(-1),
                    joint_vel_err.unsqueeze(-1),
                    self.joint_vel_err_last.unsqueeze(-1),
                    self.joint_vel_err_last_last.unsqueeze(-1)
                ), dim=2).view(-1, 6)   # shape: (num_envs*num_actions, 6)
                torques = self.actuator_network(self.actuator_net_input).view(self.num_envs, self.num_actions)
                
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_vel_err_last_last = torch.clone(self.joint_vel_err_last)
            self.joint_pos_err_last = torch.clone(joint_pos_err)
            self.joint_vel_err_last = torch.clone(joint_vel_err)
            
            return torques
            
        else:
            return super()._compute_torques(actions)
    