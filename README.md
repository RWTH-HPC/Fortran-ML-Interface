# Fortran-ML-Interface

## CLAIX-2023 Environment
The code has been tested on CLAIX-2023 using the following software modules:
```bash
module purge
module load intel-compilers/2023.1.0
module load impi/2021.9.0
export FC=mpiifort
export CC=mpiicc
export CXX=mpiicpc
module load imkl/2023.1.0
#module load Score-P/8.4

module load Python/3.11.3
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load CMake
```

## Installation
Please use the provided `install.sh` script.
It will automatically install all of the required dependencies as described below.
It also defines the variable `${FORTRAN_ML_ROOT}` to refer to the top level root directory of this repository.

### Python Virtual Environment
Create a Python virtual environment with the following commands:
```bash
python3 -m venv venv-fortran-ml-py3113
source ./venv-fortran-ml-py3113/bin/activate

pip install --no-cache-dir --no-binary :all: mpi4py==3.1.6
pip install Cython
pip install tensorflow==2.17.0
pip install tf-keras==2.17

pip install matplotlib
pip install scipy

pip install scorep
```
Notes: 
  * Disable cache and binary to make sure that mpi4y gets build with the correct MPI implementation from the module system.
  * newer version of mpi4py (4.0.1) has some binary incompatibility
  * Cython is needed to compile PhyDLL later on.
  * matplotlib is only used to visualize results of the testprograms 


### h5fortran
* (https://github.com/geospace-code/h5fortran/blob/main/Install.md)
```bash
cd extern
git clone https://github.com/geospace-code/h5fortran.git
mkdir BUILD && cd BUILD
mkdir INSTALL
cmake .. -DCMAKE_INSTALL_PREFIX=${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL
cmake --build . -j && cmake --install .
```
Note: h5fortran will automatically install its own HDF5 installation (v1.14.3) - no need to load HDF5 module on the cluster!

### TensorFlow (C API)
The AIxeleratorService relies on the C/C++ API of TensorFlow.
Thus, it requires an TensorFlow installation outside Python providing the `libtensorflow.so`.
For now we will download a precompiled version from the official TensorFlow releases.
```bash
TF_VERSION="2.17.0"
mkdir -p ./extern/tensorflow
cd ./extern/tensorflow
wget "https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}/libtensorflow-gpu-linux-x86_64.tar.gz"
tar -xzvf libtensorflow-gpu-linux-x86_64.tar.gz
rm libtensorflow-gpu-linux-x86_64.tar.gz
```
Note: You may change the version of TensorFlow here. 
In this case make sure to also install the same version beforehand into your Python virtual environment and update all paths that refer to the TensorFlow installation and Python version below accordingly!

### AIxeleratorService
```bash
cd extern/aixeleratorservice
mkdir BUILD && cd BUILD
cmake .. -DWITH_TORCH=OFF -DWITH_TENSORFLOW=ON -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_ROOT}/venv-fortran-ml-py3113/lib/python3.11/site-packages/tensorflow/
```

### PhyDLL
```bash
make BUILD_DIR=${FORTRAN_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON ENABLE_FORTRAN=ON

LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/phydll/BUILD/lib:$LD_LIBRARY_PATH PYTHONPATH=$PYTHONPATH:${FORTRAN_ML_ROOT}/extern/phydll/src/python make BUILD_DIR=${FORTRAN_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON ENABLE_FORTRAN=ON TEST_VERBOSE=ON install
```

## Build ML Interface
```bash
mkdir BUILD && cd BUILD
cmake .. -Dh5fortran_ROOT=${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_ROOT}/venv-fortran-ml-py3113/lib/python3.11/site-packages/tensorflow/ -DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_NCSA=ON
cmake --build . && cmake --install .
```

## Build ML Interface with Score-P
The ML module can also be built with Score-P for profiling and tracing.
First create the scorep-wrappers for mpiifort and mpiicpc if they are not provided by the Score-P module from the cluster.
```bash
mkdir scorep-wrapper
scorep-wrapper --create mpiifort ./scorep-wrapper
scorep-wrapper --create mpiicpc ./scorep-wrapper
```
Afterwards built the ML module using the scorep wrappers.
```bash
mkdir BUILD-SCOREP && cd BUILD-SCOREP
PATH=$PATH:${FORTRAN_ML_ROOT}/scorep-wrapper SCOREP_WRAPPER=off SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --io=none --nomemory" SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP" cmake .. -Dh5fortran_ROOT=${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_ROOT}/venv-fortran-ml-py3113/lib/python3.11/site-packages/tensorflow/ -DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_NCSA=ON -DCMAKE_C_COMPILER=scorep-mpiicc -DCMAKE_CXX_COMPILER=scorep-mpiicpc -DCMAKE_Fortran_COMPILER=scorep-mpiifort
cmake --build . && cmake --install .
```
