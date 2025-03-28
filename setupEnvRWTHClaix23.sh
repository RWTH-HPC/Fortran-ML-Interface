#!/bin/zsh

module purge
#module load intel/2023b
module load intel-compilers/2023.1.0
module load impi/2021.9.0
#export MPICH_F90=ifx
#export MPICH_CC=icx
#export MPICH_CXX=icpx
export FC=mpiifort
export CC=mpiicc
export CXX=mpiicpc
module load imkl/2023.1.0
module load Score-P/8.4

module load HDF5/1.14.0
#module load GCC/12.3.0
#module load OpenBLAS/0.3.23

module load Python/3.11.3
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load CMake

export FORTRAN_ML_VENV=${PWD}/venv-fortran-ml-py3113
source ${FORTRAN_ML_VENV}/bin/activate

if command -v readlink >/dev/null 2>&1; then
script_name="$(readlink -f "${BASH_SOURCE:-$0}")"
export FORTRAN_ML_ROOT="$(dirname "$script_name")"
echo "FORTRAN_ML_ROOT = ${FORTRAN_ML_ROOT}"
else
    echo "Error: readlink is not available on your system. Make sure it is installed!"
fi
