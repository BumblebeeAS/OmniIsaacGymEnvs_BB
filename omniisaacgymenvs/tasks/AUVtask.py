from typing import Optional
import numpy as np
import torch

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
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
   
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self._num_observations = 19
        self._num_actions = 6

        super().__init__(name, env, offset)

        self._indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)

        # intialize tensors
        # buoyancy
        self.buoyancy = torch.zeros((self._num_envs, 3), device=self._device)
        self.buoyancy_force = torch.zeros((self._num_envs, 3), device=self._device)
        self.cob = torch.zeros((self._num_envs, 3), device=self._device)
        self.buoyancy_offset = torch.zeros((self._num_envs, 3), device=self._device)

        # damping
        # disturbance

        # controller
        self.thrust = torch.zeros((self._num_envs, 6), device=self._device)
        return

    def set_up_scene(self, scene):
        self.get_auv()
        super().set_up_scene(scene)
        # self._bbauvs = CrazyflieView(prim_paths_expr="/World/envs/.*/Crazyflie", name="crazyflie_view")

        self._bbauvs = BBAUVView(prim_path_exp="/World/envs/.*/BBAUV", name="AUV_view")
        scene.add(self._bbauvs)
        scene.add(self._bbauvs.buoyancy)
        scene.add(self._bbauvs.damping)
        scene.add(self._bbauvs.controller)
        scene.add(self._bbauvs.disturbance)

    def get_auv(self):
        # copter = Crazyflie(prim_path=self.default_zero_env_path + "/Crazyflie", name="crazyflie",
        #                    translation=self._auv_position)
        bbauv = BBAUV(prim_path=(self.default_zero_env_path + "/BBAUV"), name="bbauv_1", translation=self._auv_position)
        self._sim_config.apply_articulation_settings("bbauv", get_prim_at_path(bbauv.prim_path), self._sim_config.parse_actor_config("BBAUV"))

    def get_observations(self):
        # print("get observation")
        self.position, self.orientation = self._bbauvs.get_world_poses(clone=False)
        self.velocity = self._bbauvs.get_velocities(clone=False)

        self.obs_buf[..., 0:3] = self.position
        self.obs_buf[..., 3:7] = self.orientation
        self.obs_buf[..., 7:13] = self.velocity
        self.obs_buf[..., 13:19] = self.thrust.clone()
        observations = {
            self._bbauvs.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations
    
    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return
        # print("pre physics step")
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

            # store action
        actions = actions.clone().to(self._device)
        self.thrust = actions
        # Apply forces
        # apply bouyancy force
        self.buoyancy_force = quat_rotate(quat_conjugate(self.orientation), self.buoyancy)
        # self.buoyancy_offset = quat_rotate(self.orientation, self.cob)
        self._bbauvs.buoyancy.apply_forces_and_torques_at_pos(forces=self.buoyancy_force,
                                                              positions=self.cob,
                                                              is_global=False)
        # # apply disturbances
        # apply damping forces

        # apply controller forces
        self._bbauvs.controller.apply_forces_and_torques_at_pos(forces=self.thrust[..., 0:3],
                                                                torques=self.thrust[..., 3:6],
                                                                is_global=False)
    
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
        # copied from crazyflie example

        # pos reward
        self.target_dist = torch.sqrt(torch.square(self.initial_pos - self.position).sum(-1))
        pos_reward = 1 / (1 + self.target_dist)

        # orient reward
        ups = quat_axis(self.orientation, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        # effort reward
        effort = torch.square(self.thrust).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)

        # # spin reward
        spin = torch.square(self.velocity).sum(-1)
        spin_reward = 0.01 * torch.exp(-1.0 * spin)

        self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spin_reward) - effort_reward 

    def reset_idx(self, env_ids):
        # print("reset idx ", env_ids)

        num_resets = len(env_ids)
        # reset robots
     
        self._bbauvs.set_world_poses(positions=self.initial_pos.clone()[env_ids],
                                     orientations=self.initial_rot.clone()[env_ids],
                                     indices=env_ids)
        self._bbauvs.set_velocities(velocities=self.initial_vel[env_ids].clone(),
                                    indices=env_ids)

        # parameters for randomizing
        max_d = 0.01
        mass_mean = 30.258
        buoyancy_mean = 34.35
        var = 1
        # random mass
        masses = torch_rand_float(mass_mean - var, mass_mean + var, (num_resets, 1), self._device)
        self._bbauvs.set_body_masses(values=masses,
                                     indices=env_ids,
                                     body_indices=torch.Tensor([self._bbauvs.get_body_index("auv4_cog_link")]))
        cogs = torch_rand_float(-max_d, max_d, (num_resets, 3), device=self._device)
        self._bbauvs.set_body_coms(positions=cogs,
                                   indices=env_ids,
                                   body_indices=torch.Tensor([self._bbauvs.get_body_index("auv4_cog_link")]))

        # ramdom inertia

        # random cob
        buoyancy = torch_rand_float((buoyancy_mean - 3) * 9.81, (buoyancy_mean + 3) * 9.81, (1,num_resets), self._device)
        self.buoyancy[env_ids, 2] = buoyancy
        cobs = torch_rand_float(-max_d, max_d, (num_resets, 3), device=self._device)
        cobs[...,2] -= max_d
        self.cob[env_ids] = cobs

        # random disturbance
        # random damping
        # random damping
        # random added mass

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def is_done(self):
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)

        # reset if too far from initial
        die = torch.where(self.target_dist > 2, ones, die)

        # reset if upside down
        die = torch.where(self.orient_z < 0.0, ones, die)

        # reset due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
        # print("is done ", self.reset_buf)


