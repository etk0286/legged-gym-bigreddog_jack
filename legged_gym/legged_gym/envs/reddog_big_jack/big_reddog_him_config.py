# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob
from legged_gym.envs.base.him_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
MOTION_FILES = glob.glob('/home/csl/amp/AMP_for_hardware/datasets/anymal/expert_trot_0_5.txt')
class BigReddogHimRoughCfgJack( LeggedRobotCfg ):

    class env:
        num_envs = 1000
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 #+ 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0 # measure_heights = False
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_x = [  0.3, 0.32, 0.34,0.36, 0.38,0.4, 0.42,0.44, 0.46,0.48, 0.5,0.52, 0.54,0.56, 0.58,0.6,0.62 ] # m
        # measured_points_y = [ -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':   0.1,  # [rad]
            'FR_hip_joint':  -0.1,  # [rad]
            'RL_hip_joint':   0.1,   # [rad]
            'RR_hip_joint':  -0.1,  # [rad]

            'FL_thigh_joint':  0.4,     # [rad]
            'FR_thigh_joint':  0.4,     # [rad]
            'RL_thigh_joint':  -0.4,   # [rad]
            'RR_thigh_joint':  -0.4,   # [rad]

            'FL_calf_joint': -1.0,   # [rad]
            'FR_calf_joint': -1.0,    # [rad]
            'RL_calf_joint':  1.0,    # [rad]
            'RR_calf_joint':  1.0,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.0}  # [N*m/rad] 3
        stiffness = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':   45,  # [rad]
            'FR_hip_joint':  45,  # [rad]
            'RL_hip_joint':   45,   # [rad]
            'RR_hip_joint':  45,  # [rad]

            'FL_thigh_joint':  40,     # [rad]
            'FR_thigh_joint':  40,     # [rad]
            'RL_thigh_joint':  40,   # [rad]
            'RR_thigh_joint':  40,   # [rad]

            'FL_calf_joint': 40,   # [rad]
            'FR_calf_joint': 40,    # [rad]
            'RL_calf_joint':  40,    # [rad]
            'RR_calf_joint':  40,    # [rad]
        }
        damping = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':   1,  # [rad]
            'FR_hip_joint':  1,  # [rad]
            'RL_hip_joint':   1,   # [rad]
            'RR_hip_joint':  1,  # [rad]

            'FL_thigh_joint':  1,     # [rad]
            'FR_thigh_joint':  1,     # [rad]
            'RL_thigh_joint':  1,   # [rad]
            'RR_thigh_joint':  1,   # [rad]

            'FL_calf_joint': 1,   # [rad]
            'FR_calf_joint': 1,    # [rad]
            'RL_calf_joint':  1.0,    # [rad]
            'RR_calf_joint':  1.0,    # [rad]
        }     # [N*m*s/rad] 0.1
        # damping = {'hip': 1}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25 #  0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        use_actuator_network = False

    class commands( LeggedRobotCfg.commands ):
            curriculum = True # True
            max_curriculum = 1.5 # 2.0
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1.0, 1.0] # min max [m/s]
                lin_vel_y = [-1.0, 1.0]   # min max [m/s]
                ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
                heading = [-3.14, 3.14]

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        vertical_scale = 0.002 # m

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bigreddog2/urdf/bigreddog.urdf'
        name = "big_reddog"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class domain_rand(LeggedRobotCfg.domain_rand ):
        randomize_payload_mass = True
        payload_mass_range = [-1, 3]

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = True
        friction_range = [0.2, 2.75]
        
        randomize_restitution = False
        restitution_range = [0., 1.0]
        
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        
        randomize_kp = True
        kp_range = [0.8, 1.2]
        
        randomize_kd = True
        kd_range = [0.8, 1.2]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        
        disturbance = False
        disturbance_range = [-15.0, 15.0]
        disturbance_interval = 4
        
        # disturbance = True
        # disturbance_range = [-30.0, 30.0]
        # disturbance_interval = 8

        push_robots = True
        push_interval_s = 8   #16
        max_push_vel_xy = 1.

        delay = True

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.08
            orientation = -0.25
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.03
            action_rate = -0.03
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.18
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.34
        max_contact_force = 200. # forces above this value are penalized
        clearance_height_target = -0.20


    class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height_measurements = 1
            clip_observations = 100.
            clip_actions = 100.

class BigReddogHimRoughCfgPPOJack( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_big_reddog_himLoco_jack'
        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 50000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]
        save_interval = 100 # check for potential saves every this many iterations
        max_iterations = 10000 # number of policy updates

        min_normalized_std = [0.05, 0.02, 0.05] * 4