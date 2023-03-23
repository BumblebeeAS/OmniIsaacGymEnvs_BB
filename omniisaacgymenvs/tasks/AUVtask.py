from typing import Optional
import numpy as np
import torch

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import *

from robot.bbauv import BBAUV
from robot.views.bbauv_view import BBAUVView


class AUVTaskRL(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._auv_position = torch.tensor([0.0, 0.0, 2.0])
        self._to_NED = torch.tensor([])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 19
        self._num_actions = 6

        self._indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        super().__init__(name, env, offset)

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
        RLTask.set_up_scene()
        self._bbauvs = BBAUVView(prim_path_exp="/World/envs/.*/BBAUV", name="AUV_view")
        scene.add(self._bbauvs)
        scene.add(self._bbauvs.buoyancy)
        scene.add(self._bbauvs.damping)
        scene.add(self._bbauvs.controller)
        scene.add(self._bbauvs.disturbance)

    def get_auv(self):
        bbauv = BBAUV(prim_path=self.default_zero_env_path + "/BBAUV", name="bbauv", translation=self._auv_position)
        self._sim_config.apply_articulation_settings("bbauv", get_prim_at_path(bbauv.prim_path), self._sim_config.parse_actor_config("bbauv"))

    def get_observations(self) -> dict:
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
        # store action
        self.thrust = actions.clone().to(self._device)

        # Apply forces
        # apply bouyancy force
        self.position, self.orientation = self._bbauvs.get_world_poses(clone=False)
        self.buoyancy_force = quat_rotate(quat_conjugate(self.orientation), self.buoyancy)
        self.buoyancy_offset = quat_rotate(self.orientation, self.cob)
        self._bbauvs.buoyancy.apply_forces_and_torques_at_pos(forces=self.buoyancy_force,
                                                              positions=self.buoyancy_offset,
                                                              is_global=False)
        # apply disturbances
        # apply damping forces

        # apply controller forces
        self._bbauvs.controller.apply_forces_and_torques_at_pos(forces=self.thrust[..., 0:3],
                                                                torques=self.thrust[..., 3:6],
                                                                is_global=False)
    
    def post_reset(self):
        # store initial position and velocity
        self.initial_pos, self.initial_rot = self._bbauvs.get_world_poses()
        self.initial_vel = self._bbauvs.get_velocities()
        self.position = self.initial_pos.clone()
        self.orientation = self.initial_rot.clone()

        self.reset_idx(self._indices)

    def calculate_metrics(self):
        # copied from crazyflie example

        # pos reward
        self.target_dist = torch.sqrt(torch.square(self.initial_pos - self.base_pos).sum(-1))
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
        num_resets = len(env_ids)

        # reset robots
        self._bbauvs.set_world_poses(positions=self.initial_pos[env_ids],
                                     orientations=self.initial_rot[env_ids],
                                     indices=env_ids)
        self._bbauvs.set_velocities(velocities=self.initial_vel,
                                    indices=env_ids)
        self._bbauvs.controller.apply_forces(forces=self.thrust[env_ids],
                                             indices=env_ids,
                                             is_global=False)

        # parameters for randomizing
        max_d = 0.1
        min_mass = 34
        max_mass = 40

        # random mass
        masses = torch_rand_float(min_mass, max_mass, (num_resets, 1), self._device)
        self._bbauvs.set_body_masses(values=masses,
                                     indices=env_ids,
                                     body_indices=self._bbauvs.get_body_index("auv4_cog_link"))
        cogs = torch_rand_float(-max_d, max_d, (num_resets, 3), device=self._device)
        self._bbauvs.set_body_coms(positions=cogs,
                                   indices=env_ids,
                                   body_indices=self._bbauvs.get_body_index("auv4_cog_link"))

        # ramdom inertia

        # random cob
        buoyancy = torch_rand_float(min_mass * 9.81, max_mass * 9.81, (num_resets, 1), self._device)
        self.buoyancy[env_ids, 2] = buoyancy
        cobs = torch_rand_float(-max_d, max_d, (num_resets, 3), device=self._device)
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
        die = torch.where(self.target_dist > 1, ones, die)

        # reset if upside down
        die = torch.where(self.orient_z < 0.0, ones, die)

        # reset due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)


