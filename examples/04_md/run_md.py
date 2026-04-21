from ase.io import read
from atomchain.md import md_nve_velocity_verlet
atoms = read('AlScN.vasp')
final_atoms, dyn = md_nve_velocity_verlet(
    atoms=atoms,
    calc='mace-r2scan',
    #model_path='config.conf',
    timestep=1.0,
    steps=1000,
    temperature=300.0,
    trajectory='md.traj',
    logfile="md.log"
)
