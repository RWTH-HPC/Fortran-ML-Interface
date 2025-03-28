#!/bin/zsh

# exit the script on error
set -e

if [ -n "${FORTRAN_ML_VENV}" ]; then
    echo "Found Python virtual environment: ${FORTRAN_ML_VENV}. Will use for installation."    
else
    echo "Error: No Python virtual environment defined. Please set FORTRAN_ML_ENV to the path of your virtual environment!"
fi

if command -v readlink >/dev/null 2>&1; then
script_name="$(readlink -f "${BASH_SOURCE:-$0}")"
export FORTRAN_ML_ROOT="$(dirname "$script_name")"
echo "FORTRAN_ML_ROOT = ${FORTRAN_ML_ROOT}"
else
    echo "Error: readlink is not available on your system. Make sure it is installed!"
fi

# install dependencies

## make sure Score-P wrappers are available for Intel compilers
mkdir -p ${FORTRAN_ML_ROOT}/scorep-wrapper
if [ ! -f "${FORTRAN_ML_ROOT}/scorep-wrapper/scorep-mpiifort" ]; 
then
    scorep-wrapper --create mpiifort ./scorep-wrapper
fi
if [ ! -f "${FORTRAN_ML_ROOT}/scorep-wrapper/scorep-mpiicpc" ]; 
then
    scorep-wrapper --create mpiicpc ./scorep-wrapper
fi
if [ ! -f "${FORTRAN_ML_ROOT}/scorep-wrapper/scorep-mpiicc" ]; 
then
    scorep-wrapper --create mpiicc ./scorep-wrapper
fi

## install h5fortran
if [ ! -f "${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL/lib64/libh5fortran.a" ]; then
    echo "h5fortran not found! Installing..."
    cd ${FORTRAN_ML_ROOT}/extern/h5fortran
    mkdir -p BUILD && cd BUILD
    mkdir -p INSTALL

    cmake .. -DCMAKE_INSTALL_PREFIX=${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL

    cmake --build . -j && cmake --install .
    echo "h5fortran installation finished!"
else
    echo "h5fortran installation found! Nothing to install."
fi

## install PhyDLL
# TODO: check if PhyDLL can be built with Score-P without any issues
if [ ! -f "${FORTRAN_ML_ROOT}/extern/phydll/BUILD-SCOREP/lib/libphydll.so" ]; then
    echo "PhyDLL not found! Installing..."

    cd ${FORTRAN_ML_ROOT}/extern/phydll
    mkdir -p BUILD-SCOREP
    
    PATH=$PATH:${FORTRAN_ML_ROOT}/scorep-wrapper CC=scorep-mpiicc FC=scorep-mpiifort SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --io=none --nomemory" SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP" make BUILD_DIR=${FORTRAN_ML_ROOT}/extern/phydll/BUILD-SCOREP ENABLE_PYTHON=ON ENABLE_FORTRAN=ON

    LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/phydll/BUILD-SCOREP/lib:${LD_LIBRARY_PATH} PYTHONPATH=${FORTRAN_ML_ROOT}/extern/phydll/src/python:${PYTHONPATH} PATH=$PATH:${FORTRAN_ML_ROOT}/scorep-wrapper CC=scorep-mpiicc FC=scorep-mpiifort SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --io=none --nomemory" SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP" make BUILD_DIR=${FORTRAN_ML_ROOT}/extern/phydll/BUILD-SCOREP ENABLE_PYTHON=ON ENABLE_FORTRAN=ON TEST_VERBOSE=ON install

    echo "PhyDLL installation finished!"
else
    echo "PhyDLL installation found! Nothing to install."
fi


# install TensorFlow (C API)
TF_VERSION="2.17.0"
if [ ! -f "${FORTRAN_ML_ROOT}/extern/tensorflow/lib/libtensorflow.so" ]; then
    echo "TensorFlow not found! Installing..."
    
    mkdir -p ${FORTRAN_ML_ROOT}/extern/tensorflow
    cd ${FORTRAN_ML_ROOT}/extern/tensorflow
    wget "https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}/libtensorflow-gpu-linux-x86_64.tar.gz"

    tar -xzvf libtensorflow-gpu-linux-x86_64.tar.gz
    rm libtensorflow-gpu-linux-x86_64.tar.gz

    echo "TensorFlow installation finished!"
else
    echo "TensorFlow installation found! Nothing to install."
fi

## install AIxeleratorService
if [ ! -f "${FORTRAN_ML_ROOT}/extern/aixeleratorservice/BUILD-SCOREP/lib/libAIxeleratorService.so" ]; then
    echo "AIxeleratorService not found! Installing..."

    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=1 --nocompiler --user --io=none --nomemory"
    export SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP"  

    cd ${FORTRAN_ML_ROOT}/extern/aixeleratorservice/
    mkdir -p BUILD-SCOREP && cd BUILD-SCOREP
    PATH=$PATH:${FORTRAN_ML_ROOT}/scorep-wrapper SCOREP_WRAPPER=off cmake .. -DWITH_TORCH=OFF -DWITH_TENSORFLOW=ON -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_VENV}/lib/python3.11/site-packages/tensorflow -DCMAKE_C_COMPILER=scorep-mpiicc -DCMAKE_CXX_COMPILER=scorep-mpiicpc -DCMAKE_Fortran_COMPILER=scorep-mpiifort
    VERBOSE=1 cmake --build . -j && cmake --install .

    echo "AIxeleratorService installation finished!"
else
    echo "AIxeleratorService installation found! Nothing to install."
fi

# install Fortran-ML-Interface
if [ ! -f "${FORTRAN_ML_ROOT}/BUILD-SCOREP/lib/libmlCoupling.so" ]; then
    echo "Fortran-ML-Interface not found! Installing..."

    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=1 --compiler --user --io=none --nomemory"
    export SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP"

    cd ${FORTRAN_ML_ROOT}
    mkdir -p BUILD-SCOREP && cd BUILD-SCOREP
    PATH=$PATH:${FORTRAN_ML_ROOT}/scorep-wrapper SCOREP_WRAPPER=off cmake .. -Dh5fortran_ROOT=${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_VENV}/lib/python3.11/site-packages/tensorflow -DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_NCSA=ON -DWITH_SCOREP_MANUAL=ON -DCMAKE_C_COMPILER=scorep-mpiicc -DCMAKE_CXX_COMPILER=scorep-mpiicpc -DCMAKE_Fortran_COMPILER=scorep-mpiifort
    cmake --build . -j && cmake --install .

    echo "Fortran-ML-Interface installation finished!"
else
    echo "Fortran-ML-Interface installation found! Nothing to install."
fi


# setup LD_LIBRARY_PATH to find all installed libaries
# PhyDLL
export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/phydll/BUILD-SCOREP/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PYTHONPATH}:${FORTRAN_ML_ROOT}/extern/phydll/BUILD-SCOREP/src/python:${FORTRAN_ML_ROOT}/model/
# TensorFlow
export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/tensorflow/lib:${LD_LIBRARY_PATH}
# AIxeleratorService
export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/aixeleratorservice/BUILD-SCOREP/lib:${LD_LIBRARY_PATH}