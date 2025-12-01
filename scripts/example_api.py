from ndim_perception_fusion.scripts.simulation import PyBulletSimulation

sim = PyBulletSimulation()
sim.insert_object((1.0, 2.0, 0.2), {'yaw': 1.57, 'fixed': False})