from typing import Optional
from omni.isaac.core.robots.robot import Robot

# from omni.isaac.sensor import IMUSensor
from omni.isaac.core.utils.stage import add_reference_to_stage
import torch
import os


class BBAUV(Robot):
    def __init__(
        self,
        prim_path: str,
        name: str = "bbauv",
        usd_path: Optional[str] = None,
        position: Optional[torch.tensor] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        self._usd_path = usd_path
        if self._usd_path is None:
            # self._usd_path = os.getcwd() + "/assets/auv4_description/urdf/auv4_isaac/auv4_isaac.usd"
            self._usd_path = "/home/saber/bbb/isaac/OmniIsaacGymEnvs_BB/omniisaacgymenvs/assets/auv4_description/urdf/auv4_isaac/auv4_isaac.usd"
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )
        # imu_path = prim_path + "/auv4_damping_link/imu_sensor"
        # self.imu = IMUSensor(prim_path=imu_path,
        #                      name="imu")
