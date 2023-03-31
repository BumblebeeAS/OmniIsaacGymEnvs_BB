from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
import torch

class BBAUVView(ArticulationView):
    def __init__(
            self,
            prim_path_exp: str,
            name: Optional[str] = "BBAUVView"
    ):
        super().__init__(
            prim_paths_expr=prim_path_exp,
            name=name,
            reset_xform_properties=False
        )

        self.buoyancy = RigidPrimView(prim_paths_expr="/World/envs/.*/BBAUV/auv4_cob_link", name="cob")
        self.controller = RigidPrimView(prim_paths_expr="/World/envs/.*/BBAUV/auv4_base_link", name="baselink")
        self.damping = RigidPrimView(prim_paths_expr="/World/envs/.*/BBAUV/auv4_damping_link", name="damping")
        self.disturbance = RigidPrimView(prim_paths_expr="/World/envs/.*/BBAUV/auv4_disturbance_link", name="disturbance")

        self.started = False
        self.alpha = 0.3  # this some magic number

    def get_accelerations(self, dt,  indices, velocities, clone=None):
        '''own implementation to calculate acceleration, referenced UUV simulator
        https://github.com/uuvsimulator/uuv_simulator/blob/master/uuv_gazebo_plugins/uuv_gazebo_plugins/src/HydrodynamicModel.cc#L102
        '''

        if not self.started:
            self.prev_velocities = self.get_velocities(clone=True)
            self.filtered_accel = torch.zeros_like(self.prev_velocities)
            self.started = True
        accel = (velocities - self.prev_velocities[indices][0]) / dt
        self.prev_velocities = velocities
        self.filtered_accel = (1 - self.alpha) * self.filtered_accel + (self.alpha * accel)
        if clone:
            return self.filtered_accel.clone()
        else:
            return self.filtered_accel
