import os
import numpy as np
import tensorflow as tf
# for TensorFlow 2.16.1 and higher: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
import tf_keras as keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from tsrgan_3D import generator_tsrgan3D


def main():
    n_filters = 64
    n_channels = 3
    upsampling = 4

    cpu_list = tf.config.list_physical_devices('CPU')
    num_cpus = len(cpu_list)
    print(f"found CPUs = {cpu_list}, num_cpus = {num_cpus}")

    gpu_list = tf.config.list_physical_devices('GPU')
    num_devices = len(gpu_list)
    print(f"found GPUs = {gpu_list}, num_gpus = {num_devices}")

    # Test on CPU
    with tf.device('/CPU:0'):
        model = generator_tsrgan3D(
                num_filters=n_filters,
                channels=n_channels,
                upfactor=upsampling
            )

        model.load_weights('weights/gan_generator_step_38500.0.h5')
        #model.load_weights('weights/gan_debug_weights_2Xupsampling.h5')
        #model.load_weights('weights/gan_generator_step_35000.0_decay_32_x2.h5')
        
        #model.load_weights('weights/gan_generator_step_38500.0_decay_64_x2_gaussian.h5')

        #weights = model.get_weights()
        #print(len(weights))
        #for i, w in enumerate(weights):
        #    print(np.shape(w))
        #    if i == 1:
        #        print(f"saving weights {i} = {w}")
        #    np.save(f"TSRGAN_weights_{i}.npy", w)
        #for l in model.layers:
        #    print(l)

        #model.save("TSRGAN_3D_2X_debug.tf")
        #model.save("TSRGAN_3D_36_2X_decay.tf")
        
        model.save(f"TSRGAN_3D_36_{upsampling}X_decay_gaussian.keras")
        model.export(f"TSRGAN_3D_36_{upsampling}X_decay_gaussian.tf")

        # test inference
        np.set_printoptions(precision=16)
        print(f"Input data = ")
        input_data = np.array(
            [
                [
                    [
                        [ [ 0.1, 0.5, 0.2 ], [ 0.15, 0.6, 0.1 ] ],
                        [ [ 0.4, 0.7, 0.3 ], [ 0.8,  0.9, 0.2 ] ]
                    ],
                    [
                        [ [ 0.8,  0.2,  0.45 ], [ 0.35, 0.76, 0.98 ] ],
                        [ [ 0.23, 0.75, 0.17 ], [ 0.82, 0.48, 0.89 ] ]
                    ]
                ]
            ]
        )
        print(f"Shape of input data = {np.shape(input_data)}")
        print(f"Input data reshaped 1D = {np.reshape(input_data, (24))}")

        output_data = model(tf.cast(input_data, tf.float32))[0].numpy()
        print(f"Output data = ")
        print(output_data)
        print(f"Shape of output data = {np.shape(output_data)}")

        #quit("remove me after debug")

    # Test on GPU
    if num_devices > 0:
        with tf.device('/GPU:1'):
            model_gpu = generator_tsrgan3D(
                    num_filters=n_filters,
                    channels=n_channels,
                    upfactor=upsampling
                )

            model_gpu.load_weights('weights/gan_generator_step_38500.0.h5')  

            np.set_printoptions(precision=16)
            print(f"Input data = ")
            input_data_gpu = np.array(
                [
                    [
                        [
                            [ [ 0.1, 0.5, 0.2 ], [ 0.15, 0.6, 0.1 ] ],
                            [ [ 0.4, 0.7, 0.3 ], [ 0.8,  0.9, 0.2 ] ]
                        ],
                        [
                            [ [ 0.8,  0.2,  0.45 ], [ 0.35, 0.76, 0.98 ] ],
                            [ [ 0.23, 0.75, 0.17 ], [ 0.82, 0.48, 0.89 ] ]
                        ]
                    ]
                ]
            )
            print(f"Shape of input data = {np.shape(input_data_gpu)}")
            print(f"Input data reshaped 1D = {np.reshape(input_data_gpu, (24))}")

            output_data_gpu = model_gpu(tf.cast(input_data_gpu, tf.float32))[0].numpy()
            print(f"Output data = ")
            print(output_data_gpu)
            print(f"Shape of output data = {np.shape(output_data_gpu)}")

            #quit("remove me after debug")


            abs_err = np.subtract(output_data, output_data_gpu)
            print(f"Sum of output CPU = {np.sum(output_data)}")
            print(f"Sum of output GPU = {np.sum(output_data_gpu)}")
            print(f"rel. err = {(np.sum(output_data) - np.sum(output_data_gpu)) / np.sum(output_data)}")
            sum_err = np.sum(abs_err)
            print(f"Sum of absolute elementwise errors: {sum_err}") 

    


    # model2 = generator_tsrgan3D(
    #             num_filters=n_filters,
    #             channels=n_channels,
    #             upfactor=upsampling
    #         )
    # # load weights from numpy files
    # weights = []
    # for i in range(42):
    #     w = np.load(f"TSRGAN_weights_{i}.npy")
    #     weights.append(w) 
    # model2.set_weights(weights)
    # model2.compile()
    # model2.summary()

    # wgts_old = model.get_weights()
    # wgts_new = model2.get_weights()

    # for i in range(42):
    #     w_old = wgts_old[i]
    #     #print(f"Shape of old weights {i}: {np.shape(w_old)}")
    #     w_new = wgts_new[i]
    #     #print(f"Shape of new weights {i}: {np.shape(w_new)}")

    #     diff = np.array(w_old - w_new)
    #     print(f"Sum of absolute error for weights {i}: {np.sum(diff)}")


    # # inference
    # model2.run_eagerly = True
    # shape_x = 33
    # shape_y = 33
    # shape_z = 33
    # les_data = np.zeros((1, shape_x, shape_y, shape_z, 3))
    # sr_data = model2(tf.cast(les_data, tf.float32))[0].numpy()
    
    # model2.save("TSRGAN_3D_36_4X_decay_gaussian.keras")
    # model2.export("TSRGAN_3D_36_4X_decay_gaussian.tf")
    # keras.models.save_model(model2, "TSRGAN_3D_36_4X_decay_gaussian.pb")
    # tf.saved_model.save(model2, "TSRGAN_3D_36_4X_decay_gaussian.tflow")


if __name__ == "__main__":
    main()