# Fortran-ML-Interface



## Software Environment (CLAIX-2023)
The code has been tested on CLAIX-2023 using the following software stack:
* Intel Compilers 2021.9.0 20230302
* IntelMPI 2021.9.0 20230302
* CUDA 12.3.0
* cuDNN 8.9.7.29
* HDF5 1.14.0
* Python 3.11.3
* CMake 3.26.3

Please refer to the [`setupEnvRWTHClaix23.sh`](setupEnvRWTHClaix23.sh) for details to create a similiar environment on your system!  
It is important that your environment defines the variable `FORTRAN_ML_VENV` containing the path to the root directory of a Python virtual environment, that you will use for this project. 
This environment will automatically be created during the installation process if not already present.

## Installation
### Baseline
Please refer to the provided [`install.sh`](install.sh) script.
It will automatically download and install the following required dependencies: 
* Python virtual environment
* h5fortran
* TensorFlow (C-API) 2.17.0
* AIxeleratorService
* PhyDLL v0.2

Afterwards, the compiled Fortran ML Interface can be found in `${FORTRAN_ML_ROOT}/BUILD`, where `${FORTRAN_ML_ROOT}` refers to the top level root directory of this repository (as defined by the script).

For more information the individual installation steps are documented [here](doc/install.md).

### Score-P Instrumentation
If you want to instrument the code using Score-P for profiling/tracing, we additionally provide the [`install-scorep.sh`](install-scorep.sh) script.
This script builds the whole project and its dependencies (except h5fortran and TensorFlow) using Score-P into `${FORTRAN_ML_ROOT}/BUILD_SCOREP`.

## Mini-Apps
We provide two representative mini-apps to demonstrate how the Fortran ML Module may be integrated into real production-level CFD solvers.

### Super-Resolution-based Turbulence Model
The first mini-app demonstrates the use of a Generative Adversarial Network in the context of Large Eddy SImulation, which predicts super-resolved velocity fields, based on which the unresolved Reynolds stress tensor $\tau_{ij}$. A detailed desription of the model can be found in [L. Nista, et al., "Influence of adversarial
training on super-resolution turbulence reconstruction", Physical Review Fluids 9, 2024](https://doi.org/10.1103/PhysRevFluids.9.064601).
The mini-app will read data from a real simulation snapshot of a "Forced Homogeneous Isotropic Turbulence (FHIT)" case stored in [`data/FHIT_32x32x32_output.h5`](data), feed this data to the Fortran ML module as input for the DL model, and finally output the computed Reynolds stress tensor.

#### Source Code
The source code of this mini-app can be found in [`test/main_tsrgan.f90`](test/main_tsrgan.f90).

#### Saved Model File
Before you can run the code, you need to create a saved model file of the deep learning model, that was trained with TensorFlow.
All model related files are stored in [`model/tsrgan/`](model/tsrgan/).
The definition of the model can be found in [`tsrgan_3D.py`](model/tsrgan/tsrgan_3D.py).
The weights obtained from training are stored in [`weights/gan_generator_step_38500.0.h5`](model/tsrgan/weights).
To create the saved model file, that can be loaded by the mini-app later, you should use the provided Python script [`createGenerator.py`](model/tsrgan/createGenerator.py).
It will output the final saved model file (actually a directory in the TensorFlow case) `TSRGAN_3D_36_4X_decay_gaussian.tf/`.

#### Execution
The code expects a command line parameter `coupling_strategy_id` to determine which inference strategy is used in the backend. 
Currently supported options are {AIxeleratorService = 1, PhyDLL = 2}.
The program is MPI-parallel but each process will execute the same computational work.
So in it's current version the mini-app can only be used for weak-scaling experiments but not for strong-scaling.

To run with the AIxeleratorService simply execute:
```bash
mpirun -np <num_clients> main_tsrgan.x 1
```
where `<num_clients>` is the number of processes that execute the
If you execute the mini-app on a compute node, where GPUs are available, the AIxeleratorService will internally detect the available GPU devices and make use of them.
To control which GPU devices should be used, you may want to define `CUDA_VISIBLE_DEVICES`.

To run with PhyDLL you need to execute the mini-app in an MPMD fashion:
```bash
mpirun -np <num_clients> main_tsrgan.x 2 : -np <num_servers> python3 ${FORTRAN_ML_ROOT}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll_tsrgan.py -nphy <num_clients>
```

### CNN-based Combustion Model
The second mini-app demonstrates the use of a UNet convolutional neural network architecture, which predicts reaction rates based on progress variable input in a hydrogen combustion case. A detailed desription of the model can be found in [G. Arumapperuma, et al., "Extrapolation Performance of Convolutional Neural Network-Based Combustion Models for Large-Eddy Simulation: Influence of Reynolds Number, Filter Kernel and Filter Size", Flow, Turbulent, and Combustion, 2025](https://doi.org/10.1103/PhysRevFluids.9.064601).
The mini-app will read data from a real direct numerical simulation snapshot of a lean hydrogen flame case stored in [`data/H2_000057_prate.h5`](data), feed this data  to the Fortran ML module as input for the DL model, and finally output the predicted reaction rates.

#### Source Code
The source code of this mini-app can be found in [`test/main_cnn.f90`](test/main_cnn.f90).

#### Saved Model File
Before you can run the code, you need to create a saved model file of the deep learning model, that was trained with TensorFlow.
All model related files are stored in [`model/unet/_2D/`](model/unet/_2D/).
The definition of the model can be found in [`CNN.py`](model/unet/_2D/CNN.py).
The weights obtained from training are stored in [`Model_1251_loss_0.00012877.h5`](model/unet/_2D).
To create the saved model file, that can be loaded by the mini-app later, you should use the provided Python script [`convert.py`](model/unet/_2D/convert.py):
```bash
python3 convert.py --path ${FORTRAN_ML_ROOT}/model/unet/_2D
```
It will output the final saved model file (actually a directory in the TensorFlow case) `Model_1251_loss_0.00012877.tf`.

#### Execution
The code expects a command line parameter `coupling_strategy_id` to determine which inference strategy is used in the backend. 
Currently supported options are {AIxeleratorService = 1, PhyDLL = 2}.
The program is MPI-parallel but each process will execute the same computational work.
So in it's current version the mini-app can only be used for weak-scaling experiments but not for strong-scaling.

To run with the AIxeleratorService simply execute:
```bash
mpirun -np <num_clients> main_cnn.x 1
```
where `<num_clients>` is the number of processes that execute the
If you execute the mini-app on a compute node, where GPUs are available, the AIxeleratorService will internally detect the available GPU devices and make use of them.
To control which GPU devices should be used, you may want to define `CUDA_VISIBLE_DEVICES`.

To run with PhyDLL you need to execute the mini-app in an MPMD fashion:
```bash
mpirun -np <num_clients> main_cnn.x 2 : -np <num_servers> python3 ${FORTRAN_ML_ROOT}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.py -nphy <num_clients> -dimUNet 2
```


## Citation
If you are using the Fortran ML Interface in your own work, please cite the following paper:
```
@article{
  title={{E}fficient and {S}calable {AI}xeleration of {R}eactive {CFD} {S}olvers {C}oupled with {D}eep {L}earning {I}nference on {H}eterogeneous {A}rchitectures},
  author={Fabian Orland and Ludovico Nista and Nick Kocher and Joris Vanvinckenroye and Heinz Pitsch and Christian Terboven},
  journal={2025 International Conference on High Performance Computing in Asia-Pacific Region Workshop Proceedings (HPC Asia 2025 Workshops)},
  year={2025},
  doi = {10.1145/3703001.3724386}
}
```
