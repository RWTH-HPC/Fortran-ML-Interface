import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
#import scorep

import h5py

def main():
    num_filters = 1
    kernel_size = (2, 2, 2)
    padding = "valid" # valid -> no padding, same -> zero padding


    model = models.Sequential()
    model.add(layers.Conv3D(num_filters, kernel_size, padding=padding, input_shape=(3, 3, 3, 3), name='myconv', use_bias=False))
    weights = model.get_layer('myconv').get_weights()
    print(weights)
    print(weights[0].shape)
    # shape of weights (x, y, z, channels, num_filters)
    kernel_weights = np.array(
        [
            [
                [ [1, 1, 1], [2, 2, 2] ],
                [ [3, 3, 3], [4, 4, 4] ]
            ],
            [
                [ [5, 5, 5], [6, 6, 6] ],
                [ [7, 7, 7], [8, 8, 8] ]
            ]
        ]
    ).reshape((2,2,2,3,1))

    print(kernel_weights)
    model.get_layer('myconv').set_weights([ kernel_weights ])

    model.summary()
    model.save("testConvolution3D-multichannel.keras")
    model.export("testConvolution3D-multichannel.tf")

    # shape of input data = (3,3,3,3)
    input_data = np.array(
        [
            [
                [
                    [ [ 1, 28, 55], [ 2, 29, 56], [ 3, 30, 57] ],
                    [ [ 4, 31, 58], [ 5, 32, 59], [ 6, 33, 60] ],
                    [ [ 7, 34, 61], [ 8, 35, 62], [ 9, 36, 63] ]
                ],
                [
                    [ [10, 37, 64], [11, 38, 65], [12, 39, 66] ],
                    [ [13, 40, 67], [14, 41, 68], [15, 42, 69] ],
                    [ [16, 43, 70], [17, 44, 71], [18, 45, 72] ]
                ],
                [
                    [ [19, 46, 73], [20, 47, 74], [21, 48, 75] ],
                    [ [22, 49, 76], [23, 50, 77], [24, 51, 78] ],
                    [ [25, 52, 79], [26, 53, 80], [27, 54, 81] ]
                ]
            ]
        ]
    )

    with h5py.File(f"./cnn3d-multichannel-input-py.h5", "w") as file:
        file.create_dataset("/input/all-channels", data=np.transpose(input_data))
        file.create_dataset("/input/channel1", data=np.transpose(input_data[:,:,:,0]))
        file.create_dataset("/input/channel2", data=np.transpose(input_data[:,:,:,1]))
        file.create_dataset("/input/channel3", data=np.transpose(input_data[:,:,:,2]))

    print(f"Reshaped 1D input data = {input_data.reshape(np.prod(input_data.shape))}")

    print(f"input[0,0,0,0,0] = {input_data[0,0,0,0,0]}")
    print(f"input[0,0,0,0,1] = {input_data[0,0,0,0,1]}")
    print(f"input[0,0,0,1,0] = {input_data[0,0,0,1,0]}")
    print(f"input[0,0,1,0,0] = {input_data[0,0,1,0,0]}")
    print(f"input[0,1,0,0,0] = {input_data[0,1,0,0,0]}")

    output_data = model(input_data)
    print(output_data)
    print(np.shape(output_data))

    print(f"Reshaped 1D output data = {np.reshape(output_data, (np.prod(output_data.shape)))}")

    #print(model.get_weights())
    #print(np.shape(model.get_weights()))

    print(f"out[0,0,0,0,0] = {output_data[0,0,0,0,0]}")
    print(f"out[0,0,0,1,0] = {output_data[0,0,0,1,0]}")
    print(f"out[0,0,1,0,0] = {output_data[0,0,1,0,0]}")
    print(f"out[0,0,1,1,0] = {output_data[0,0,1,1,0]}")
    print(f"out[0,1,0,0,0] = {output_data[0,1,0,0,0]}")
    print(f"out[0,1,0,1,0] = {output_data[0,1,0,1,0]}")
    print(f"out[0,1,1,0,0] = {output_data[0,1,1,0,0]}")
    print(f"out[0,1,1,1,0] = {output_data[0,1,1,1,0]}")

if __name__ == "__main__":
    print(tf.__version__)
    main()  

# 3D Conv example
#1) 1*1 + 2*2 + 3*4 + 4*5 + 5*10 + 6*11 + 7*13 + 8*14 
#2) 1*2 + 2*3 + 3*5 + 4*6 + 5*11 + 6*12 + 7*14 + 8*15 
#3) 1*4 + 2*5 + 3*7 + 4*8 + 5*13 + 6*14 + 7*16 + 8*17 
#4) 1*5 + 2*6 + 3*8 + 4*9 + 5*14 + 6*15 + 7*17 + 8*18 
#5) 1*10 + 2*11 + 3*13 + 4*14 + 5*19 + 6*20 + 7*22 + 8*23 
#6) 1*11 + 2*12 + 3*14 + 4*15 + 5*20 + 6*21 + 7*23 + 8*24 
#7) 1*13 + 2*14 + 3*16 + 4*17 + 5*22 + 6*23 + 7*25 + 8*26 
#8) 1*14 + 2*15 + 3*17 + 4*18 + 5*23 + 6*24 + 7*26 + 8*27 