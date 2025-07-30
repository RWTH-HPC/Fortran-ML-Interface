#!/bin/zsh

module purge

module load intel-compilers/2023.1.0
module load impi/2021.9.0
module load intel/2024a
export FC=mpiifx
export CC=mpiicx
export CXX=mpiicpx

module load HDF5/1.14.5

module load Python/3.12.3
export FORTRAN_ML_PYVERSION="3.12" # keep consistent with loaded module

module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load CMake

module load Score-P/9.2

export FORTRAN_ML_VENV=${PWD}/venv-fortran-ml-py3123
source ${FORTRAN_ML_VENV}/bin/activate

if command -v readlink >/dev/null 2>&1; then
script_name="$(readlink -f "${BASH_SOURCE:-$0}")"
export FORTRAN_ML_ROOT="$(dirname "$script_name")"
echo "FORTRAN_ML_ROOT = ${FORTRAN_ML_ROOT}"
else
    echo "Error: readlink is not available on your system. Make sure it is installed!"
fi
