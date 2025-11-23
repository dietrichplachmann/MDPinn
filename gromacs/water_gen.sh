# make water molecule coordinate file
gmx pdb2gmx -f water.pdb -o water_processed.gro -water spce

# make box
gmx editconf -f water_processed.gro -o water_box.gro -c -d 1.0 -bt cubic

# solvate
gmx solvate -cp water_box.gro -cs spc216.gro -o solv.gro -p topol.top
