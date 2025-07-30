#!/bin/zsh

# exit the script on error
set -e

# if [ -n "${FORTRAN_ML_VENV}" ]; then
#     echo "Found Python virtual environment: ${FORTRAN_ML_VENV}. Will use for installation."    
# else
#     echo "Error: No Python virtual environment defined. Please set FORTRAN_ML_ENV to the path of your virtual environment!"
# fi

if [ ! -f "${FORTRAN_ML_VENV}/bin/activate" ]; then
    echo "No Python virtual environment found. Installing now!"

    python3 -m venv "${FORTRAN_ML_VENV}"
    source "${FORTRAN_ML_VENV}/bin/activate"

    pip install --no-cache-dir --no-binary :all: mpi4py==3.1.6
    pip install Cython
    pip install tensorflow==2.17.0
    pip install tf-keras==2.17.0

    pip install matplotlib
    pip install scipy

    pip install scorep
else
    echo "Found Python virtual environment! Nothing to install. Activating environment now!"
    source "${FORTRAN_ML_VENV}/bin/activate"
fi

if command -v readlink >/dev/null 2>&1; then
script_name="$(readlink -f "${BASH_SOURCE:-$0}")"
export FORTRAN_ML_ROOT="$(dirname "$script_name")"
echo "FORTRAN_ML_ROOT = ${FORTRAN_ML_ROOT}"
else
    echo "Error: readlink is not available on your system. Make sure it is installed!"
fi

# install dependencies

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
if [ ! -f "${FORTRAN_ML_ROOT}/extern/phydll/BUILD/lib/libphydll.so" ]; then
    echo "PhyDLL not found! Installing..."

    cd ${FORTRAN_ML_ROOT}/extern/phydll
    mkdir -p BUILD
    
    make BUILD_DIR=${FORTRAN_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON ENABLE_FORTRAN=ON

    LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/phydll/BUILD/lib:${LD_LIBRARY_PATH} PYTHONPATH=${FORTRAN_ML_ROOT}/extern/phydll/src/python:${PYTHONPATH} make BUILD_DIR=${FORTRAN_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON ENABLE_FORTRAN=ON TEST_VERBOSE=ON install

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
if [ ! -f "${FORTRAN_ML_ROOT}/extern/aixeleratorservice/BUILD/lib/libAIxeleratorService.so" ]; then
    echo "AIxeleratorService not found! Installing..."

    cd ${FORTRAN_ML_ROOT}/extern/aixeleratorservice/
    mkdir -p BUILD && cd BUILD
    cmake .. -DWITH_TORCH=OFF -DWITH_TENSORFLOW=ON -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_VENV}/lib/python${FORTRAN_ML_PYVERSION}/site-packages/tensorflow
    cmake --build . -j && cmake --install .

    echo "AIxeleratorService installation finished!"
else
    echo "AIxeleratorService installation found! Nothing to install."
fi

# install Fortran-ML-Interface
if [ ! -f "${FORTRAN_ML_ROOT}/BUILD/lib/libmlCoupling.so" ]; then
    echo "Fortran-ML-Interface not found! Installing..."

    cd ${FORTRAN_ML_ROOT}
    mkdir -p BUILD && cd BUILD
    cmake .. -Dh5fortran_ROOT=${FORTRAN_ML_ROOT}/extern/h5fortran/BUILD/INSTALL -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_VENV}/lib/python${FORTRAN_ML_PYVERSION}/site-packages/tensorflow -DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_NCSA=ON
    cmake --build . -j && cmake --install .

    echo "Fortran-ML-Interface installation finished!"
else
    echo "Fortran-ML-Interface installation found! Nothing to install."
fi

# setup LD_LIBRARY_PATH to find all installed libaries
# PhyDLL
export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/phydll/BUILD/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${PYTHONPATH}:${FORTRAN_ML_ROOT}/extern/phydll/BUILD/src/python:${FORTRAN_ML_ROOT}/model/
# TensorFlow
export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/tensorflow/lib:${LD_LIBRARY_PATH}
# AIxeleratorService
export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/aixeleratorservice/BUILD/lib:${LD_LIBRARY_PATH}
