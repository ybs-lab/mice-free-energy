while getopts t:p:s:n: flag
do
    case "${flag}" in
        t) tmp=${OPTARG};;
        p) phase=${OPTARG};;
        s) seed=${OPTARG};;
        n) time=${OPTARG};;
    esac
done

mkdir ${seed}
cd    ${seed}

mkdir log
mkdir dump
mkdir restart
mkdir data
mkdir colvar
mkdir plumed

##########################################################################

cat > "start_temp_press.lmp" << EOF
echo both
variable temperature equal ${tmp}
variable pressure equal 1.
variable tempDamp equal .1 # approx 0.1 ps
variable pressureDamp equal 10.0
variable seed world ${seed} 



variable        side equal 8
variable        numAtoms equal 2048
variable        mass equal 26.981539
units metal
lattice 	    fcc 4.049
region          box block 0 \${side} 0 \${side} 0 \${side}
atom_style full
create_box      1 box
create_atoms    1 random \${numAtoms} ${seed} box
mass            1 \${mass}
change_box      all triclinic
pair_style  eam/fs
pair_coeff  * * ../../../potentials/Al1.eam.fs Al
variable        out_freq equal 100
variable        dump_freq equal 2500
variable        restart_freq equal 200000
neigh_modify    delay 10 every 1
timestep        0.002 # According to Frenkel and Smit is 0.001
thermo          \${out_freq}
thermo_style    custom step temp pe ke press density vol enthalpy atoms lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz
restart         \${restart_freq} restart/restart.\${temperature}
log log/log.lammps append

minimize 0 1.0e-8 1000 10000
reset_timestep 0

dump 	 myDump all custom \${dump_freq} dump/dump\${temperature}.lammpstrj id type x y z
dump_modify  myDump append no

# NVT
variable kenergy equal ke
variable penergy equal pe
variable pres equal press
variable tempera equal temp
variable dense equal density
variable entha equal enthalpy 
variable enthaperatom equal enthalpy/\${numAtoms}

fix myat1 all ave/time 10 1 10 v_kenergy v_penergy v_pres v_tempera v_dense v_entha v_enthaperatom file data/energy_\${temperature}.dat

fix             1 all nve
fix             2 all temp/csvr \${temperature} \${temperature} \${tempDamp} \${seed}
velocity        all create \${temperature} \${seed} dist gaussian
run             10000
unfix           1
unfix           2 

# fix             2 all nph &
#                 x \${pressure} \${pressure} \${pressureDamp} &
#                 y \${pressure} \${pressure} \${pressureDamp} &
#                 z \${pressure} \${pressure} \${pressureDamp} &
#                 xy 0.0 0.0 \${pressureDamp} &
#                 yz 0.0 0.0 \${pressureDamp} &
#                 xz 0.0 0.0 \${pressureDamp} &
#                 couple xyz
fix             1 all plumed plumedfile ../plumedAl.dat outfile plumed/plumed\${temperature}.out
fix             2 all nph iso \${pressure} \${pressure} \${pressureDamp}
fix             3 all temp/csvr \${temperature} \${temperature} \${tempDamp} \${seed}
fix             4 all momentum 10000 linear 1 1 1 angular

variable 	steps equal \${ns}*500000 #1 ns
run           	\${steps} 

write_data	data.final

EOF

#########################################################################

mpirun -np 8 lmp_mpi -v ns ${time} -in start_temp_press.lmp -screen none

