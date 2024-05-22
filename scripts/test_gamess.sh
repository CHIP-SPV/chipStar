#!/bin/bash
set -ex

source /etc/profile.d/modules.sh &> /dev/null
module use ~/modulefiles

cd /home/pvelesko/GAMESS/gamess_libcchem_hip

export IGC_EnableDPEmulation=1
export OverrideDefaultFP64Settings=1

module purge
module load llvm/18.0
module load HIP/chipStar/testing
module load hdf5/1.14.4.2
module load openmpi/4.1.1
module load HipSolver/2024.05.15-807efe2 

export JSON_ROOT=$PWD/../json
export MATHLIB_ROOT=$HIP_DIR
export GPU_BOARD=Intel
export OMP_NUM_THREADS=1

export MPI_CXX=mpic++
rm -rf build && mkdir build && cd build
cmake -DMPI_ROOT=$MPI_ROOT   -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++  -DJSON_ROOT=$JSON_ROOT -DHDF5_ROOT=$HDF5_ROOT  -DHIP=True -DMATHLIB_ROOT=$MATHLIB_ROOT -DHIPSOLVER_ROOT=$HIPSOLVER_ROOT -DBUILD_RIMP2=0 -DMAGMA=False -DGPU_BOARD=$GPU_BOARD -DMKLROOT=$MKLROOT ../
make -j $(nproc)
cd ..

export IGC_ForceOCLSIMDWidth=16
#export ZE_AFFINITY_MASK=0.0

orterun --mca btl vader,self,tcp -np 2 --bind-to core --map-by ppr:2:node ./build/exess ./inputs/json_inputs/scf/w1.json 2>&1 | tee gamess_output.txt

# Extract the floating point number from the output file
energy=$(grep "Final energy is:" gamess_output.txt | awk '{print $4}')
correct_energy=-74.9612532341291

# Define the tolerance value
tol=0.0001

# Calculate the absolute difference between the energy and the correct energy
diff=$(echo "$energy - $correct_energy" | bc | tr -d -)

# Compare the absolute difference with the tolerance value
comparison=$(echo "$diff < $tol" | bc)

if [ $comparison -eq 1 ]
then
    echo "The energy value is within the tolerance."
    exit 0
else
    echo "The energy value is not within the tolerance."
    exit 1
fi