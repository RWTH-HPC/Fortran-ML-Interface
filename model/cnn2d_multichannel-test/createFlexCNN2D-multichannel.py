import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
#import scorep

def main():
    num_filters = 1
    kernel_size = (2, 2)
    padding = "valid" # valid -> no padding, same -> zero padding


    model = models.Sequential()
    model.add(layers.Conv2D(num_filters, kernel_size, padding=padding, input_shape=(3, 3, 2), name='myconv', use_bias=False))
    weights = model.get_layer('myconv').get_weights()
    print(weights)
    print(weights[0].shape)
    kernel_weights = np.array(
        [
            [
                [0,1], [1,2]
            ],
            [
                [2,3], [3,4]
            ]
        ]
    ).reshape((2,2,2,1))
    model.get_layer('myconv').set_weights([ kernel_weights ])

    model.summary()
    model.save("testConvolution2D-multichannel.tf")

    input_data = np.array(
            [
                [
                    [
                        [0,1], [1,2], [2,3]
                    ],
                    [
                        [3,4], [4,5], [5,6]
                    ],
                    [
                        [6,7], [7,8], [8,9]
                    ] 
                ]
            ]
    )

    #tf.profiler.experimental.start('tf-test-log')
    #with tf.profiler.experimental.Profile('tf-log-dir'):
    #with scorep.instrumenter.enable():
    output_data = model(input_data)
    #tf.profiler.experimental.stop()

    print(output_data)
    print(model.get_weights())

    #runOptions = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    #runConfig = tf.compat.v1.ConfigProto(run_options=runOptions)
    #runConfigSer = [int(i) for i in runConfig.SerializeToString()]
    #print(runConfigSer)


if __name__ == "__main__":
    print(tf.__version__)
    main()  
