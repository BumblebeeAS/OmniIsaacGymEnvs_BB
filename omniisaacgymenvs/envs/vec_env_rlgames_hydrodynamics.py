from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
import torch


class VecEnvHydrodynamics(VecEnvRLGames):
    def step(self, actions):
        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(actions=actions, reset_buf=self._task.reset_buf)

        actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()

        self._task.pre_physics_step(actions)

        for _ in range(self._task.control_frequency_inv):
            self._task.apply_hydrodynamics()
            self._world.step(render=self._render)
            self.sim_frame_count += 1
            self._task.get_physics_states()

        self._obs, self._rew, self._resets, self._extras = self._task.post_physics_step()

        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device), reset_buf=self._task.reset_buf)

        self._states = self._task.get_states()
        self._process_data()

        obs_dict = {"obs": self._obs, "states": self._states}

        return obs_dict, self._rew, self._resets, self._extras