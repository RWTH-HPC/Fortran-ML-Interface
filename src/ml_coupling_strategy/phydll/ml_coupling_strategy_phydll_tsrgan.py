import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
MPI.Init()

from pyphydll.pyphydll import PhyDLL
import numpy as np
import tensorflow as tf
import os
# for TensorFlow 2.16.1 and higher: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
import tf_keras as keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# import TSRGAN
from tsrgan.tsrgan_3D import generator_tsrgan3D

from timeit import default_timer as timer

import scorep.user

import nvtx

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-nphy", "--NpPHY", type=int, default=0)
args = parser.parse_args()


config_3D_TSRGAN = {
    "weights": "/work/rwth0792/fortran-ml-interface/model/tsrgan/weights/gan_generator_step_38500.0.h5",
    "data_shape": {
        1:  (1, 33, 33, 33, 3),    # only for test app
        4:  (1, 32, 32, 64, 3),    #  1 node  with  4 MPI procs
        8:  (1, 32, 32, 32, 3),    #  2 nodes with  8 MPI procs
        16: (1, 16, 32, 32, 3),    #  4 nodes with 16 MPI procs
        32: (1, 16, 32, 16, 3),    #  8 nodes with 32 MPI procs
        64: (1, 16, 16, 16, 3),    # 16 nodes with 64 MPI procs
        128: (1, 8, 16, 16, 3),    # 16 nodes with 64 MPI procs
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
            print(f"PhyDLL: my device id = {my_device_id}")
            tf.config.set_visible_devices(gpu_list[my_device_id], 'GPU')


        config = config_3D_TSRGAN

        # import DL model TSRGAN
        upsampling = 4
        model = generator_tsrgan3D(num_filters=64, channels=3, upfactor=upsampling)
        model.load_weights(config["weights"])

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
                        num_phy_procs = int(dll.get_field_size() / (input_shape[1]*upsampling * input_shape[2]*upsampling * input_shape[3]*upsampling * input_shape[4]))
                        print(f"PhyDLL: num physical procs = {num_phy_procs}")

                        if input_shape[3] == 1:
                            dl_shape = (input_shape[0] * num_phy_procs, input_shape[1], input_shape[2], input_shape[4])
                        else:
                            dl_shape = (input_shape[0] * num_phy_procs, input_shape[1], input_shape[2], input_shape[3], input_shape[4])
                        dl_input = np.reshape(input_field[0:(field_size*num_phy_procs)], dl_shape)
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
                    with scorep.user.region("input-copy"):
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
        MPI.Finalize()


if __name__ == "__main__":
    main()