import CNN
import argparse
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf2
# for TensorFlow 2.16.1 and higher: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
#from tensorflow import keras
import tf_keras as keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
#from keras import optimizers
import var_train_losses
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert model checkpoints stored in .h5 format to .tf format")
    parser.add_argument('--path', type=str)

    args = parser.parse_args()
    #tf2.disable_v2_behavior()
    #tf2.disable_eager_execution()
    for file in os.listdir(args.path):
        if file.endswith(".h5"):
            if not file.startswith('data'):
                print(file)

                #try:
                #    model = CNN.CNN()
                #    model.load_weights(args.path + '/' + file)
                #except ValueError:
                #    model = CNN.CNN_2()
                #    model.load_weights(args.path + '/' + file)
                loss = var_train_losses.custom_h2o_loss_with_weights()
                model = keras.models.load_model(args.path + '/' + file,
                                                   custom_objects={'custom_loss': loss, 'CustomPReLU': CNN.CustomPReLU})
                adamOptimizer = keras.optimizers.Adam(learning_rate=0.0001,
                                                clipnorm=1.)
                model.compile(optimizer=adamOptimizer, loss=loss, metrics=['mae'],
                              experimental_run_tf_function=False)
                model.run_eagerly = True
                model.summary()
                print(model.inputs[0].dtype)
                input_data = np.ones((1, 2048, 2048, 1), dtype=np.float32)
                print(model(input_data))
                input_data = np.ones((1, 512, 1024, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 1024, 512, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 512, 512, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 256, 512, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 512, 256, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 256, 256, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 128, 256, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 256, 128, 1), dtype=np.float32)
                #print(model(input_data))
                input_data = np.ones((1, 128, 128, 1), dtype=np.float32)
                #print(model(input_data))
                #tf.compat.v1.disable_eager_execution()
                #tf.saved_model.save(model, args.path + '/' + os.path.splitext(file)[0] + '.tf')
                model.save(args.path + '/' + os.path.splitext(file)[0] + '.tf')


