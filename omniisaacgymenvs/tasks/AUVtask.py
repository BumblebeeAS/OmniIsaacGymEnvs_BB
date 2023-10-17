import math
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.utils.torch.maths import *
from omni.isaac.core.utils.torch.rotations import *

from omniisaacgymenvs.robots.articulations.bbauv import BBAUV
from omniisaacgymenvs.robots.articulations.views.bbauv_view import BBAUVView

from omniisaacgymenvs.robots.articulations.crazyflie import Crazyflie
from omniisaacgymenvs.robots.articulations.views.crazyflie_view import CrazyflieView


class AUVTaskRL(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._auv_position = torch.tensor([0.0, 0.0, 2.0])
        self._ball_position = torch.tensor([0.0, 0.0, 2.0])

        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 18
        self._num_actions = 6

        self.counter = 0

        self.extras = {} # for wandb plotting

        super().__init__(name, env, offset)

        self._indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        """initialize tensors"""
        self.position = torch.zeros((self._num_envs, 3), device=self._device)
        self.orientation = torch.zeros((self._num_envs, 4), device=self._device)
        self.x_vec = torch.zeros((self._num_envs,3), device=self._device)
        self.y_vec = torch.zeros((self._num_envs,3), device=self._device)
        self.z_vec = torch.zeros((self._num_envs,3), device=self._device)
        self.roll = torch.zeros((self._num_envs,1), device=self._device)
        self.pitch = torch.zeros((self._num_envs,1), device=self._device)
        self.yaw = torch.zeros((self._num_envs,1), device=self._device)
        self.velocity_global = torch.zeros((self._num_envs, 6), device=self._device)
        self.instantaneous_velocity = torch.zeros((self._num_envs, 6), device=self._device)
        self.instantaneous_acceleration = torch.zeros((self._num_envs, 6), device=self._device)

        # buoyancy
        self.buoyancy = torch.zeros((self._num_envs, 3), device=self._device)
        self.buoyancy[..., 2] = 349
        self.buoyancy_force = torch.zeros((self._num_envs, 3), device=self._device)
        self.cob = torch.zeros((self._num_envs, 3), device=self._device)
        self.buoyancy_offset = torch.zeros((self._num_envs, 3), device=self._device)

        # damping
        self.linear_damping_diag = torch.tensor(
            (self._num_envs * [[3.8, 4.4, 34.8, 0.131, 0.201, 0.077]]), device=self._device
        )
        self.quadratic_damping_diag = torch.tensor(
            (self._num_envs * [[38.7, 54.6, 78.9, 0.342, 0.465, 0.451]]), device=self._device
        )
        self.added_mass_diag = torch.tensor(
            (self._num_envs * [[12.19, 17.44, 26.47, 0.15, 0.26, 0.19]]), device=self._device
        )

        self.linear_damping = torch.zeros((self._num_envs, 6, 6), device=self._device)
        self.quadratic_damping = torch.zeros((self._num_envs, 6, 6), device=self._device)
        self.added_mass = torch.zeros((self._num_envs, 6, 6), device=self._device)

        # disturbance

        # controller
        self.thrust = torch.zeros((self._num_envs, 6), device=self._device)
        self.prev_effort = torch.zeros((self._num_envs, 6), device=self._device)
        self.force_multiplier = 35
        self.torque_multiplier = 5

        if self.position.size(dim=0) == 4:
            self.x_plot = torch.zeros(( self._max_episode_length -2, 4), device=self._device)
            self.y_plot = torch.zeros(( self._max_episode_length -2, 4), device=self._device)
            self.z_plot = torch.zeros(( self._max_episode_length -2, 4), device=self._device)
            self.xang_plot = torch.zeros((self._max_episode_length - 2,4), device=self._device)
            self.yang_plot = torch.zeros((self._max_episode_length - 2,4), device=self._device)
            self.zang_plot = torch.zeros((self._max_episode_length - 2,4), device=self._device)
            self.plot_step = 0
        

        # self.model_testing = torch.jit.load("/home/saber/bbb/isaac/OmniIsaacGymEnvs_BB/omniisaacgymenvs/scripts/runs/BBAUV/nn/last_BBAUV_ep_1575_rew_4119.8457.pt")
        return

    def set_up_scene(self, scene):
        self.get_auv()
        self.get_target()
        super().set_up_scene(scene)
        self._bbauvs = BBAUVView(prim_path_exp="/World/envs/.*/BBAUV", name="AUV_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")
        scene.add(self._bbauvs)
        scene.add(self._bbauvs.buoyancy)
        scene.add(self._bbauvs.damping)
        scene.add(self._bbauvs.controller)
        scene.add(self._bbauvs.disturbance)
        scene.add(self._balls)

        self.ball_positions, _ = self._balls.get_world_poses(clone=True)

    def get_auv(self):
        bbauv = BBAUV(prim_path=(self.default_zero_env_path + "/BBAUV"), name="bbauv_1", translation=self._auv_position)
        self._sim_config.apply_articulation_settings(
            "bbauv", get_prim_at_path(bbauv.prim_path), self._sim_config.parse_actor_config("BBAUV")
        )

    def get_target(self):
        radius = 0.05
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color,
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )
        ball.set_collision_enabled(False)

    def get_observations(self):
        pos_error = self.position - self.ball_positions
        if self.position.size(dim=0) == 4 and self.plot_step < self._max_episode_length- 2:
            self.x_plot[self.plot_step] = pos_error[...,0]
            self.y_plot[self.plot_step] = pos_error[...,1]
            self.z_plot[self.plot_step] = pos_error[...,2]
            # euler_x, euler_y, euler_z = get_euler_xyz_wrapped(self.orientation)
            self.xang_plot[self.plot_step] = self.roll
            self.yang_plot[self.plot_step] = self.pitch
            self.zang_plot[self.plot_step] = self.yaw
            self.plot_step += 1
        if self.position.size(dim=0) == 4 and self.plot_step == self._max_episode_length -2:
            self.plot_graph()
            # print(self.xang_plot[-1]
            self.plot_step +=1
        pos_error = quat_rotate(quat_conjugate(self.orientation), pos_error)
        self.obs_buf[..., 0:3] = torch.clamp(pos_error.clone(), -3, 3)
        self.obs_buf[..., 3] = self.roll.clone() / torch.pi 
        self.obs_buf[..., 4] = self.pitch.clone() / torch.pi
        self.obs_buf[..., 5] = self.yaw.clone() / torch.pi
        self.obs_buf[..., 6:12] = torch.clamp(self.instantaneous_velocity.clone(), -3, 3)
        self.obs_buf[..., 12:19] = self.thrust.clone()


        # self.obs_buf[..., 0:3] = torch.clamp(pos_error.clone(), -3, 3)
        # self.obs_buf[..., 3:6] = self.x_vec.clone()
        # self.obs_buf[..., 6:9] = self.y_vec.clone()
        # self.obs_buf[..., 9:12] = self.z_vec.clone()
        # self.obs_buf[..., 12:18] = torch.clamp(self.instantaneous_velocity.clone(), -3, 3)
        # self.obs_buf[..., 18:24] = self.thrust.clone()
        # print('observation', self.obs_buf[0])
        observations = {self._bbauvs.name: {"obs_buf": self.obs_buf}}
        return observations

    def get_physics_states(self):
        self.position, self.orientation = self._bbauvs.get_world_poses(clone=True)
        self.x_vec = quat_axis(self.orientation, 0)
        self.y_vec = quat_axis(self.orientation, 1)
        self.z_vec = quat_axis(self.orientation, 2)
        self.roll, self.pitch, self.yaw = get_euler_xyz_wrapped(self.orientation)
        self.velocity_global = self._bbauvs.get_velocities(clone=True)
        self.instantaneous_velocity[..., 0:3] = quat_rotate(
            quat_conjugate(self.orientation), self.velocity_global[..., 0:3]
        )
        self.instantaneous_velocity[..., 3:6] = quat_rotate(
            quat_conjugate(self.orientation), self.velocity_global[..., 3:6]
        )
        acceleration = self._bbauvs.get_accelerations(
            dt=self.dt, indices=self._indices, velocities=self.velocity_global.clone()
        )
        self.instantaneous_acceleration[..., 0:3] = quat_rotate(
            quat_conjugate(self.orientation), acceleration[..., 0:3]
        )
        self.instantaneous_acceleration[..., 3:6] = quat_rotate(
            quat_conjugate(self.orientation), acceleration[..., 3:6]
        )

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return

        # reset environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

            # store action
        actions = actions.clone().to(self._device)
        self.thrust = actions
        # print("action", self.thrust[0])

    def apply_hydrodynamics(self):
        """Apply forces"""
        # apply bouyancy force
        self.buoyancy_force = quat_rotate(quat_conjugate(self.orientation), self.buoyancy)
        # self.buoyancy_offset = quat_rotate(self.orientation, self.cob)
        self._bbauvs.buoyancy.apply_forces_and_torques_at_pos(
            forces=self.buoyancy_force, positions=self.cob, is_global=False
        )
        # # apply disturbances

        # apply damping forces
        lin_damping_forces = torch.bmm(self.linear_damping, self.instantaneous_velocity.reshape(self._num_envs, 6, 1))
        quad_damping_forces = torch.bmm(
            self.quadratic_damping,
            (
                self.instantaneous_velocity.clone().reshape(self._num_envs, 6, 1)
                * torch.abs(self.instantaneous_velocity.clone().reshape(self._num_envs, 6, 1))
            ),
        )
        added_mass_forces = torch.bmm(self.added_mass, self.instantaneous_acceleration.reshape(self._num_envs, 6, 1))
        # print(quad_damping_forces[0])
        self.total_damping = -(lin_damping_forces + quad_damping_forces + added_mass_forces).reshape(self._num_envs, 6)
        # self.total_damping = - (lin_damping_forces + added_mass_forces).reshape(self._num_envs, 6)

        self._bbauvs.damping.apply_forces_and_torques_at_pos(
            forces=self.total_damping[..., 0:3], torques=self.total_damping[..., 3:6], is_global=False
        )

        # apply controller forces
        # print(self.thrust)
        torque = self.thrust[..., 3:6]
        torque[...,2] *= 2
        self._bbauvs.controller.apply_forces_and_torques_at_pos(
            forces=(self.thrust[..., 0:3] * self.force_multiplier),
            torques=(torque * self.torque_multiplier),
            is_global=False,
        )

    def post_reset(self):
        # print("post reset")

        # store initial position and velocity
        self.initial_pos, self.initial_rot = self._bbauvs.get_world_poses(clone=False)
        self.initial_vel = self._bbauvs.get_velocities(clone=False)
        self.position = self.initial_pos.clone()
        self.orientation = self.initial_rot.clone()

        self.reset_idx(self._indices)
        # print(self.initial_rot)

    def calculate_metrics(self):
        # print("calculate metrics")
        self.counter += 1

        # pos reward
        self.target_dist = torch.sqrt(torch.square(self.ball_positions - self.position).sum(-1))
        pos_reward = 2 / (1 + self.target_dist)

        # orient reward
        ups = quat_axis(self.orientation, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        forward = quat_axis(self.orientation, 0)
        self.orient_x = forward[..., 0]
        forward_reward = torch.clamp(forward[..., 0], min=-1.0, max=1.0)
        forward_reward = (forward_reward + 1.0) / 2.0

        # effort reward
        effort = torch.square(self.thrust).sum(-1)
        effort_reward = 0.01 * torch.exp(-0.5 * effort)

        # effort change reward
        effort_change = torch.square(self.thrust - self.prev_effort).sum(-1)
        effort_change_reward = 0.1 * torch.exp(-0.5 * effort_change)

        # # spin reward
        spin = torch.square(self.velocity_global[:, 3:]).sum(-1)
        spin_reward = 0.1 * torch.exp(-1.0 * spin)

        # self.rew_buf[:] = pos_reward - effort_reward
        self.rew_buf[:] = pos_reward + 10 * up_reward * pos_reward + forward_reward * pos_reward
        """new rewards function"""
        # position reward
        # self.target_dist = torch.sqrt(torch.square(self.initial_pos - self.position).sum(-1))
        # pos_reward = 1 / (1 + self.target_dist)

        # orientation reward

    def reset_idx(self, env_ids):
        noise = 0
        num_resets = len(env_ids)
        # reset robots
        root_pos = self.ball_positions.clone()
        curriculum = (min(1, self.counter/20000)) if (self.test == False) else 0.5
        # print(curriculum)

        root_pos[env_ids, 0] += torch_rand_float(-0.5, 0.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.5, 0.5, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.5, 0.5, (num_resets, 1), device=self._device).view(-1)

        # curriculum = min(1, self.counter/500000)

        euler_angles = torch.zeros((self._num_envs, 3), device=self._device)
        euler_angles[env_ids,2] += torch_rand_float(-3.14 /2, 3.14 /2, (num_resets, 1), device=self._device).view(-1)
        # euler_angles[env_ids,2] += torch.clamp(torch.normal(torch.zeros((num_resets,1), device=self._device), curriculum).view(-1), -3.14, 3.14)
        root_rot = euler_angles_to_quats(euler_angles, device=self._device)
        self._bbauvs.set_world_poses(
            positions=root_pos[env_ids], orientations=root_rot[env_ids], indices=env_ids
        )
        self._bbauvs.set_velocities(velocities=self.initial_vel[env_ids].clone(), indices=env_ids)

        # parameters for randomizing
        buoyancy_offset = 0.05
        max_d = noise * 0.02  # 0.02
        mass_mean = 30.258
        buoyancy_mean = 34.35
        mass_var = noise * 0.3  # 0.3
        damping_var = noise * 0.1  # 0.1

        # random mass
        masses = torch_rand_float(mass_mean - mass_var, mass_mean + mass_var, (num_resets, 1), self._device)
        self._bbauvs.set_body_masses(
            values=masses, indices=env_ids, body_indices=torch.Tensor([self._bbauvs.get_body_index("auv4_cog_link")])
        )
        cogs = torch_rand_float(-max_d, max_d, (num_resets, 3), device=self._device)
        # self._bbauvs.set_body_coms(positions=cogs,
        #                            indices=env_ids,
        #                            body_indices=torch.Tensor([self._bbauvs.get_body_index("auv4_cog_link")]))

        # ramdom inertia

        # random cob
        buoyancy = torch_rand_float(
            (buoyancy_mean - mass_var) * 9.81, (buoyancy_mean + mass_var) * 9.81, (1, num_resets), self._device
        )
        self.buoyancy[env_ids, 2] = buoyancy
        cobs = torch_rand_float(-max_d, max_d, (num_resets, 3), device=self._device)
        cobs[..., 2] += buoyancy_offset
        self.cob[env_ids] = cobs

        # random disturbance

        # random damping
        self.linear_damping[env_ids] = torch_rand_mat(-damping_var, damping_var, (num_resets, 6, 6), self._device)
        self.linear_damping[env_ids] += self.linear_damping[env_ids].clone().transpose(-2, -1)  # this ensures symetry
        self.linear_damping[env_ids] += self.linear_damping_diag[env_ids].clone().diag_embed()

        self.quadratic_damping[env_ids] = torch_rand_mat(-damping_var, damping_var, (num_resets, 6, 6), self._device)
        self.quadratic_damping[env_ids] += (
            self.quadratic_damping[env_ids].clone().transpose(-2, -1)
        )  # this ensures symetry
        self.quadratic_damping[env_ids] += self.quadratic_damping_diag[env_ids].clone().diag_embed()

        # random added mass
        # self.added_mass[env_ids] = torch_rand_mat(-damping_var, damping_var, (num_resets, 6, 6), self._device)
        # self.added_mass[env_ids] += self.added_mass[env_ids].clone().transpose(-2, -1)  # this ensures symetry
        # self.added_mass[env_ids] += self.added_mass_diag[env_ids].clone().diag_embed()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def is_done(self):
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # reset if too far from initial
        die = torch.where(self.target_dist > 2, ones, die)

        # reset if upside down
        die = torch.where(self.orient_z < 0.0, ones, die)

        # die = torch.where(self.orient_x < 0.0, ones, die)
        # reset due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
        # print("is done ", self.reset_buf)

    def plot_graph(self):
        print("plotting graph")

        self.x_plot.to(device='cpu')
        self.y_plot.to(device='cpu')
        self.z_plot.to(device='cpu')
        self.xang_plot.to(device='cpu')
        self.yang_plot.to(device='cpu')
        self.zang_plot.to(device='cpu')
        steps = np.arange(0, self._max_episode_length -2) * 0.05
        plt.figure(figsize=(12, 8))

        plotx = torch.transpose(self.x_plot.cpu(), 0 ,1)
        ploty = torch.transpose(self.y_plot.cpu(), 0 ,1)
        plotz = torch.transpose(self.z_plot.cpu(), 0 ,1)
        plotr = torch.transpose(self.xang_plot.cpu(), 0 ,1)
        plotp = torch.transpose(self.yang_plot.cpu(), 0 ,1)
        plotyw = torch.transpose(self.zang_plot.cpu(), 0 ,1)
        # plot = torch.transpose(self.x_plot, 0 ,1)

        plt.subplot(3, 2, 1)
        plt.plot(steps,plotx[0], color='r')
        plt.plot(steps,plotx[1], color='g')
        plt.plot(steps,plotx[2], color='b')
        plt.plot(steps,plotx[3], color='y')
        plt.xlabel('Time(s)')
        plt.ylabel('X Error(m)')
        plt.title('X Error')

        plt.subplot(3, 2, 2)
        plt.plot(steps,ploty[0], color='r')
        plt.plot(steps,ploty[1], color='g')
        plt.plot(steps,ploty[2], color='b')
        plt.plot(steps,ploty[3], color='y')
        plt.xlabel('Time(s)')
        plt.ylabel('Y Error(m)')
        plt.title('Y Error')

        plt.subplot(3, 2, 3)
        plt.plot(steps,plotz[0], color='r')
        plt.plot(steps,plotz[1], color='g')
        plt.plot(steps,plotz[2], color='b')
        plt.plot(steps,plotz[3], color='y')       
        plt.xlabel('Time(s)')
        plt.ylabel('Z Error(m)')
        plt.title('Z Error')

        plt.subplot(3, 2, 4)
        plt.plot(steps,plotr[0], color='r')
        plt.plot(steps,plotr[1], color='g')
        plt.plot(steps,plotr[2], color='b')
        plt.plot(steps,plotr[3], color='y')        
        plt.xlabel('Time(s)')
        plt.ylabel('Roll Error(rad)')
        plt.title('Roll Error')

        plt.subplot(3, 2, 5)
        plt.plot(steps,plotp[0], color='r')
        plt.plot(steps,plotp[1], color='g')
        plt.plot(steps,plotp[2], color='b')
        plt.plot(steps,plotp[3], color='y')        
        plt.xlabel('Time(s)')
        plt.ylabel('Pitch Error(rad)')
        plt.title('Pitch Error')

        plt.subplot(3, 2, 6)
        plt.plot(steps,plotyw[0], color='r')
        plt.plot(steps,plotyw[1], color='g')
        plt.plot(steps,plotyw[2], color='b')
        plt.plot(steps,plotyw[3], color='y')        
        plt.xlabel('Time(s)')
        plt.ylabel('Yaw Error(rad)')
        plt.title('Yaw Error')

        plt.suptitle('Error Plots')
        plt.tight_layout()  # Adjust layout to prevent overlapping labels

        # Save the plots as PNG files
        plt.savefig(self._name + '.jpg')
        # plt.show()


@torch.jit.script
def torch_rand_mat(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int,int], str) -> Tensor
    # return (upper - lower) * torch.rand(*shape, device=device) + lower
    return torch.zeros(*shape, device=device) + upper

@torch.jit.script
def get_euler_xyz_wrapped(q):
    qw, qx, qy, qz = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll , pitch , yaw