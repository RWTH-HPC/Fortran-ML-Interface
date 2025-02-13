print(f"PHYDLL TEST 1", flush=True)
import time
sleep_time = 1
print(f"PHYDLL sleeping for {sleep_time} seconds now!")
time.sleep(sleep_time)
print(f"PHYDLL awake! Going to initialize MPI now!")
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
print(f"PHYDLL TEST 2", flush=True)
MPI.Init()
print(f"PHYDLL TEST 3 --> MPI Init done", flush=True)

from pyphydll.pyphydll import PhyDLL
import numpy as np
import tensorflow as tf
import os
# for TensorFlow 2.16.1 and higher: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
import tf_keras as keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from unet._2D.CNN import CNN_2, CustomPReLU
from unet._2D import var_train_losses

# import TSRGAN
from tsrgan.tsrgan_3D import generator_tsrgan3D

from timeit import default_timer as timer

import scorep.user

import nvtx 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-nphy", "--NpPHY", type=int, default=0)
parser.add_argument("-dimUNet", "--dimUNet", type=int, default=2)
args = parser.parse_args()

config_2D_test = {
    "model": "/work/rwth0792/fortran-ml-interface/model/cnn2d-test/testConvolution2D.tf",
    "input_shape": (1, 3, 3, 1, 1),
    "output_shape": (1, 2, 2, 1, 1)
}

config_3D_test = {
    "model": "/work/rwth0792/fortran-ml-interface/model/cnn3d-test/testConvolution3D.tf",
    "input_shape": (1, 3, 3, 3, 1),
    "output_shape": (1, 2, 2, 2, 1)
}

config_3D_test_multichannel = {
    "model": "/work/rwth0792/fortran-ml-interface/model/cnn3d-multichannel-test/testConvolution3D-multichannel.tf",
    "input_shape": (1, 3, 3, 3, 3),
    "output_shape": (1, 2, 2, 2, 1)
}

config_2D_omega_unet = {
    "model": "/hpcwork/rwth0792/CIAO_UNet_laminar_hydrogen_flame/ai4s-2024-CIAO-UNet/common/models/OMEGA_Model_1257_loss_0.00004648.tf",
    "weights": "/hpcwork/rwth0792/CIAO_UNet_laminar_hydrogen_flame/ai4s-2024-CIAO-UNet/common/models/weights/OMEGA_Model_1257_loss_0.00004648.h5",
    "data_shape": {
          1: (1, 2044, 2044, 1, 1),
          8: (1, 512, 1024, 1, 1),
         16: (1, 512,  512, 1, 1),
         32: (1, 256,  512, 1, 1),
         64: (1, 256,  256, 1, 1),
        128: (1, 128,  256, 1, 1)
    }
}

config_2D_omega_unetpp = {
    "model": "/hpcwork/rwth0792/CIAO_UNet_laminar_hydrogen_flame/ai4s-2024-CIAO-UNet/common/models/OMEGA_UNETpp_1066_loss_0.00009053.tf",
    "weights": "/hpcwork/rwth0792/CIAO_UNet_laminar_hydrogen_flame/ai4s-2024-CIAO-UNet/common/models/weights/OMEGA_UNETpp_1066_loss_0.00009053.h5",
    "data_shape": {
          1: (1, 2044, 2044, 1, 1),
          8: (1, 512, 1024, 1, 1),
         16: (1, 512,  512, 1, 1),
         32: (1, 256,  512, 1, 1),
         64: (1, 256,  256, 1, 1),
        128: (1, 128,  256, 1, 1)
    }
}

config_3D_omega_unet = {
    "model": "/work/rwth0792/fortran-ml-interface/model/unet/_3D_new/Model_170_loss_10.81598186.tf",
    "data_shape": {
          1: (1, 384, 192, 72, 1),
          8: (1, 96, 96, 72, 1),
         16: (1, 48, 96, 72, 1),
         32: (1, 48, 96, 36, 1),
         64: (1, 48, 48, 36, 1),
        128: (1, 24, 48, 36, 1)
    }
}

config_3D_TSRGAN = {
    "weights": "/work/rwth0792/fortran-ml-interface/model/tsrgan/weights/gan_generator_step_38500.0.h5",
    "data_shape": {
        1: (1, 33, 33, 33, 3)    
    }
}

print(f"PHYDLL TEST 4 before main", flush=True)

def main():
    print(f"PHYDLL TEST 5 in main", flush=True)
    with scorep.user.region("main"):
        dll = PhyDLL()

        dll.init(instance="dl")

        comm = dll.get_local_mpi_comm()
        my_local_rank = comm.Get_rank()
        #my_global_rank = MPI.COMM_WORLD.Get_rank()
        num_dl_procs = comm.Get_size()
        print(f"PhyDLL: number of DL processes = {num_dl_procs}")

        gpus_per_node = 4 # on CLAIX23 4x Nvidia H100 GPUs per node
        cpu_list = tf.config.list_physical_devices('CPU')
        num_cpus = len(cpu_list)
        print(f"PhyDLL found CPUs = {cpu_list}, num_cpus = {num_cpus}")
        gpu_list = tf.config.list_physical_devices('GPU')
        num_devices = len(gpu_list)
        print(f"PhyDLL found GPUs = {gpu_list}, num_gpus = {num_devices}")

        if num_devices > 0:
            my_device_id = my_local_rank % gpus_per_node
            tf.config.set_visible_devices(gpu_list[my_device_id], 'GPU')

        #config = config_2D_test
        #config = config_3D_test
        #config = config_3D_test_multichannel
        #config = config_2D_omega_unet
        #config = config_2D_omega_unetpp
        #config = config_3D_TSRGAN

        if args.dimUNet == 2:
            config = config_2D_omega_unetpp

        if args.dimUNet == 3:
            config = config_3D_omega_unet

        # import DL model
        #model = CNN_2()
        #model.load_weights(config["weights"])

        loss = var_train_losses.custom_h2o_loss_with_weights() 
        model = keras.models.load_model(config["model"], custom_objects={'custom_loss': loss, 'CustomPReLU': CustomPReLU})

        # TSRGAN
        #model = generator_tsrgan3D(num_filters=64, channels=3, upfactor=4)
        #model.load_weights(config["weights"])

        num_fields = 1
        dll.define_dl(num_fields)

        phy_count, _ = dll.get_field_counts()
        print(f"PhyDLL: phy_count = {phy_count}")

        print(f"PhyDLL determining num_phys_procs based on setup information now!", flush=True)

        num_phys_procs = args.NpPHY
        print(f"PhyDLL determined num_phys_procs = {num_phys_procs}", flush=True)
        input_shape = config["data_shape"][num_phys_procs]

        phy_fields = {}
        dl_fields = {}

        cpl_freq = 1
        output_freq = 1
        ite = 0
        
        file_timings = open(f"phydll-inference-rank-{my_local_rank}.dat", "w")
        with scorep.user.region("coupling-loop"):
            while dll.is_phy_signal():
                with scorep.user.region(f"coupling-step-{ite}"):
                    # receive fields from physical solver
                    with scorep.user.region("dll-receive"):
                        phy_fields = dll.recv()
                    phy_keys = list(phy_fields.keys())

                    with scorep.user.region("input-copy"):
                        input_field = phy_fields[phy_keys[0]]
                        print(f"Shape of input_field = {np.shape(input_field)}", flush=True)
                        # PhyDLL will return the size of the all received fields together (at least on DL ranks)
                        #field_size = dll.get_field_size()
                        field_size = np.prod(input_shape)
                        print(f"PhyDLL: field_size = {field_size}", flush=True)
                        num_phy_procs = int(dll.get_field_size() / field_size)
                        print(f"PhyDLL: num physical procs = {num_phy_procs}")

                        if input_shape[3] == 1:
                            dl_shape = (input_shape[0] * num_phy_procs, input_shape[1], input_shape[2], input_shape[4])
                        else:
                            dl_shape = (input_shape[0] * num_phy_procs, input_shape[1], input_shape[2], input_shape[3], input_shape[4])
                        dl_input = np.reshape(input_field, dl_shape)
                        #print(f"PhyDLL: dl_input = {dl_input}", flush=True)
                        print(f"PhyDLL: shape of dl_input = {dl_input.shape}", flush=True)

                    print(f"PhyDLL: Inference start.", flush=True)
                    with nvtx.annotate("model-predict", color="green"):
                        pred_start = timer()
                        with scorep.user.region("model-predict"):
                            dl_output = model.predict(dl_input)
                        pred_end = timer()
                    print(f"PhyDLL: Inference end. Duration: {pred_end - pred_start} sec.", flush=True)
                    file_timings.write(f"{pred_end - pred_start}\n")

                    #print(f"PhyDLL: dl_output = {dl_output}", flush=True)
                    with scorep.user.region("output-copy"):
                        output_shape = np.shape(dl_output)
                        output_size = np.prod(output_shape)
                        print(f"PHYDLL DEBUG 1", flush=True)
                        single_output_size = np.prod(output_shape[1:])
                        dl_output = np.reshape(dl_output, (output_size)) 

                        print(f"PHYDLL DEBUG 2", flush=True)
                        output_field = np.zeros(len(input_field))
                    
                        output_field[:] = dl_output[:]

                        dl_fields["DL_output_field_0"] = output_field

                    #dl_fields["DL_output_field_0"] = np.array(dl_output)
                    print(f"PHYDLL DEBUG 3", flush=True)
                    #print(f'PhyDLL: dl_fields_0 = {dl_fields["DL_output_field_0"]}', flush=True)
                    with scorep.user.region("dll-send"):
                        dll.send(dl_fields)

                    ite += 1

        file_timings.close()
        dll.finalize()
        #MPI.COMM_WORLD.Barrier()
        print(f"No Barrier anymore!", flush=True)
        MPI.Finalize()


if __name__ == "__main__":
    main()