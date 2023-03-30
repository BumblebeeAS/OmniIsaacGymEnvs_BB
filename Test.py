from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.core.utils.rotations import quat_to_rot_matrix

import numpy as np
import torch

from omniisaacgymenvs.robots.articulations.views.bbauv_view import BBAUVView
from omniisaacgymenvs.robots.articulations.bbauv import BBAUV

from omni.isaac.core.utils.torch.rotations import *


my_world = World(stage_units_in_meters=1)
my_world.scene.add_default_ground_plane()
define_prim("/World/envs/env_0")
num_env = 9

cloner = GridCloner(spacing=2)
cloner.define_base_env("/World/envs")
target_paths = cloner.generate_paths("/World/envs/env", num_env)
position_offsets = np.array([[0, 0, 1]] * num_env)

BBAUV(prim_path="/World/envs/env_0/BBAUV",usd_path=(os.getcwd() + "/omniisaacgymenvs/assets/auv4_description/urdf/auv4_isaac/auv4_isaac.usd"), name="bbauv_1")

cloner.clone(
    source_prim_path="/World/envs/env_0",
    prim_paths=target_paths,
    position_offsets=position_offsets,
    replicate_physics=True
)

bbauvs = BBAUVView(prim_path_exp="/World/envs/.*/BBAUV", name="AUV_view")
my_world.scene.add(bbauvs)
my_world.scene.add(bbauvs.buoyancy)
# my_world.scene.add(bbauvs.damping)
my_world.scene.add(bbauvs.controller)

my_world.reset()
# bbauvs.set_body_coms(positions=np.array([[[1, 1, 1]]]), indices=[0])
bbauvs.set_body_masses(values=[42], indices=[1], body_indices=bbauvs.get_body_index("auv4_cog_link"))
bbauvs.set_body_coms(positions=np.array([0, 0, 0.5]), indices=[0], body_indices=bbauvs.get_body_index("auv4_cog_link"))

for j in range(2000):
# while True:
    forces = np.array([[0, 0, 347]] * num_env)
    pos, rot = bbauvs.buoyancy.get_world_poses()
    offset = np.array([[0, 0 if i % 2 else 0, 0.2 if i % 2 else 0] for i in range(num_env)])
    for i in range(num_env):
        rotmat = quat_to_rot_matrix(rot[i])
        # print(np.linalg.inv(rotmat))
        # print(np.transpose(forces))
        forces[i] = np.transpose(np.linalg.inv(rotmat) @ np.transpose(forces[i]))
        if i == 5:
            offset[i] = np.transpose(rotmat @ np.transpose(offset[i]))
        # forces[i] = quat_rotate(rot[i], forces[i])
    bbauvs.buoyancy.apply_forces_and_torques_at_pos(forces=forces, positions=offset, is_global=False)
    print(bbauvs.get_accelerations(dt=0.1))
    my_world.step(render=True)

simulation_app.close()
