from typing import Optional
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import os


class BBAUV(Robot):
    def __init__(self, 
                 prim_path: str,
                 name: str = "bbauv",
                 usd_path: Optional[str] = None,
                 position: Optional[np.ndarray] = None,
                 translation: Optional[np.ndarray] = None,
                 orientation: Optional[np.ndarray] = None) -> None:
        self._usd_path = usd_path
        if self._usd_path is None:
            self._usd_path = os.getcwd() + "/assets/auv4_description/urdf/auv4_isaac/auv4_isaac.usd"
            # self._usd_path = "/home/saber/bbb/isaac/assets/auv4_description/urdf/auv4_isaac/auv4_isaac.usd"
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )
