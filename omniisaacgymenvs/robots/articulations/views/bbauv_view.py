from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


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
