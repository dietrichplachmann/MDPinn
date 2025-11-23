# energy minimization
gmx grompp -f em.mdp -c solv.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em

# NVT
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt

# NPT
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -deffnm npt

# production
gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr
gmx mdrun -deffnm prod
