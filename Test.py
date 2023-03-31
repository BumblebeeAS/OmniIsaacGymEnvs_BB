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
from omni.isaac.core.utils.torch.maths import *

import os

my_world = World(stage_units_in_meters=1, backend="torch"   )
my_world.scene.add_default_ground_plane()
define_prim("/World/envs/env_0")
num_env = 9

cloner = GridCloner(spacing=2)
cloner.define_base_env("/World/envs")
target_paths = cloner.generate_paths("/World/envs/env", num_env)
position_offsets = np.array([[0, 0, 1]] * num_env)

BBAUV(prim_path="/World/envs/env_0/BBAUV", usd_path=(os.getcwd() + "/omniisaacgymenvs/assets/auv4_description/urdf/auv4_isaac/auv4_isaac.usd"), name="bbauv_1")

cloner.clone(
    source_prim_path="/World/envs/env_0",
    prim_paths=target_paths,
    position_offsets=position_offsets,
    replicate_physics=True
)

bbauvs = BBAUVView(prim_path_exp="/World/envs/.*/BBAUV", name="AUV_view")
my_world.scene.add(bbauvs)
my_world.scene.add(bbauvs.buoyancy)
my_world.scene.add(bbauvs.damping)
my_world.scene.add(bbauvs.controller)

my_world.reset()
# bbauvs.set_body_coms(positions=np.array([[[1, 1, 1]]]), indices=[0])
bbauvs.set_body_masses(values=torch.tensor([42.]), indices=[1], body_indices=torch.tensor([bbauvs.get_body_index("auv4_cog_link")]))
bbauvs.set_body_coms(positions=torch.tensor([0, 0, 0.5]), indices=[0], body_indices=torch.tensor([bbauvs.get_body_index("auv4_cog_link")]))

for j in range(2000):
# while True:
    # bouyancy force
    bouyancy_forces = torch.tensor([[0., 0., 347.]] * num_env)
    pos, rot = bbauvs.buoyancy.get_world_poses()
    vel = bbauvs.get_velocities()
    vel_local = torch.zeros((num_env, 6))
    vel_local[..., 0:3] = quat_rotate(quat_conjugate(rot), vel[..., 0:3])
    vel_local[..., 3:6] = quat_rotate(quat_conjugate(rot), vel[..., 3:6])
    print(vel_local[3])
    offset = torch.tensor([[0, 0 if i % 2 else 0, 0.2 if i % 2 else 0] for i in range(num_env)])
    bouyancy_forces = quat_rotate(quat_conjugate(rot), bouyancy_forces) 
    bbauvs.buoyancy.apply_forces_and_torques_at_pos(forces=bouyancy_forces, positions=offset, is_global=False)

    bbauvs.controller.apply_forces(forces=torch.tensor([20., 0., 0.]), indices=[3], is_global=True)
    bbauvs.damping.apply_forces_and_torques_at_pos(torques=torch.tensor([0., 0., 5]), indices=[3])

    my_world.step(render=True)

simulation_app.close()
