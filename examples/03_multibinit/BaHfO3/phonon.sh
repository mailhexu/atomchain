# For MULTIBINIT with unit cell (5 atoms), use ndim 1 1 1
# This means phonopy will NOT create a supercell (uses the 5-atom unit cell directly)
# IMPORTANT: ndim must match the structure size that ncell expects!
mlphonon -m mb ./BaHfO3_ref.vasp --model_path ./BaHfO3_config.conf --ndim  2 2 2
