import sys
import os  
import numpy as np
import tensorflow as tf  
# for TensorFlow 2.16.1 and higher: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
import tf_keras as keras 
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from keras import backend as K
from keras import Input, optimizers, layers, regularizers
from keras.models import Model
from keras.layers import Activation, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate, Average
import h5py 


###########################################################
## f_CNN algorithm according to Fig. 4 - U-NET STRUCTURE ##
##         \bar(c) -> \bar(\omega_dot}) 	         ##
###########################################################

class CustomPReLU(keras.layers.Layer):
    def __init__(self, alpha_initializer='zeros', **kwargs):
        super(CustomPReLU, self).__init__(**kwargs)
        self.alpha_initializer = keras.initializers.get(alpha_initializer)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(1,), initializer=self.alpha_initializer, name='alpha')
        super(CustomPReLU, self).build(input_shape)

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = self.alpha * (inputs - tf.abs(inputs)) * 0.5
        return pos + neg

    def get_config(self):
        config = super(CustomPReLU, self).get_config()
        config.update({'alpha_initializer': keras.initializers.serialize(self.alpha_initializer)})
        return config

def encoder_block(inputs, filters):
    conv = Conv2D(filters, kernel_size=(3, 3), padding='same')(inputs)
    norm = BatchNormalization()(conv)
    prelu = CustomPReLU()(norm)
    return prelu

def decoder_block(inputs, skip_connections, filters):
    upsample = UpSampling2D((2, 2))(inputs)
    skip_concat = Concatenate()([upsample] + skip_connections)
    conv = Conv2D(filters, kernel_size=(3, 3), padding='same')(skip_concat)
    norm = BatchNormalization()(conv)
    prelu = CustomPReLU()(norm)
    return prelu

# Model definition
def CNN(level_input=4): # Based on UNet++ by Zongwei Zhou, et. al.
    img_shape = (None, None, 1)    
    inputs = Input(shape=img_shape)
    
    # Encoder block
    x00 = encoder_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x00)
    x10 = encoder_block(pool1, 64)
    
    if level_input >= 2:
        pool2 = MaxPooling2D(pool_size=(2, 2))(x10)
        x20 = encoder_block(pool2, 128)
    
    if level_input >= 3:
        pool3 = MaxPooling2D(pool_size=(2, 2))(x20)
        x30 = encoder_block(pool3, 256)
    
    if level_input == 4:
        pool4 = MaxPooling2D(pool_size=(2, 2))(x30)
        x40 = encoder_block(pool4, 512)
    
    # Decoder block with Dense convolution skip pathways
    if level_input == 4:
        x31 = decoder_block(x40, [x30], 256)
        x21 = encoder_block(Concatenate()([x20, UpSampling2D((2, 2))(x30)]), 128)
        x22 = decoder_block(x31, [x20, x21], 128)
        x11 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x20), x10]), 64)
        x12 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x21), x11, x10]), 64)
        x13 = decoder_block(x22, [x10, x11, x12], 64)
        x01 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x10), x00]), 32)
        x02 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x11), x01, x00]), 32)
        x03 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x12), x02, x01, x00]), 32)
        x04 = decoder_block(x13, [x00, x01, x02, x03], 32)
        output1 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x01)
        output2 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x02)
        output3 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x03)
        output4 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x04)
        prune_output = Average()([output1, output2, output3, output4]) # UNet++L4
    
    elif level_input == 3:
        x21 = encoder_block(Concatenate()([x20, UpSampling2D((2, 2))(x30)]), 128)
        x11 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x20), x10]), 64)
        x12 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x21), x11, x10]), 64)
        x01 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x10), x00]), 32)
        x02 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x11), x01, x00]), 32)
        x03 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x12), x02, x01, x00]), 32)
        output1 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x01)
        output2 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x02)
        output3 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x03)
        prune_output = Average()([output1, output2, output3]) # UNet++L3
    
    elif level_input == 2:
        x01 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x10), x00]), 32)
        x11 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x20), x10]), 64)
        x02 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x11), x01, x00]), 32)
        output1 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x01)
        output2 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x02)
        prune_output = Average()([output1, output2]) # UNet++L2
    
    elif level_input == 1:
        x01 = encoder_block(Concatenate()([UpSampling2D((2, 2))(x10), x00]), 32)
        output1 = Conv2D(1, kernel_size=(1, 1), activation='tanh', padding='same')(x01)
        prune_output = output1 # UNet++L1
    
    model = Model(inputs=inputs, outputs=prune_output)

    return model

def CNN_2():
    img_shape = (None, None, 1)  # Definition of the initial shape (1: single channel)
    inputs = Input(shape=img_shape)  # bar{c} Fig.4
    conv1 = layers.Conv2D(64, 3, padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = CustomPReLU()(conv1)
    conv1 = layers.Conv2D(64, 3, padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = CustomPReLU()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = CustomPReLU()(conv2)
    conv2 = layers.Conv2D(128, 3, padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = CustomPReLU()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = CustomPReLU()(conv3)
    conv3 = layers.Conv2D(256, 3, padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = CustomPReLU()(conv3)
    conv3 = layers.UpSampling2D(size=(2, 2))(conv3)

    up4 = layers.concatenate([conv3, conv2])
    conv4 = layers.Conv2DTranspose(128, 3, padding='same')(up4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = CustomPReLU()(conv4)
    conv4 = layers.Conv2DTranspose(128, 3, padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)  # conv ou crop
    conv4 = CustomPReLU()(conv4)
    conv4 = layers.Conv2DTranspose(128, 1, padding='same')(conv4)
    conv4 = layers.UpSampling2D(size=(2, 2))(conv4)

    up5 = layers.concatenate([conv4, conv1])
    conv5 = layers.Conv2DTranspose(64, 3, padding='same')(up5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = CustomPReLU()(conv5)
    conv5 = layers.Conv2DTranspose(64, 3, padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)  # conv ou crop
    conv5 = CustomPReLU()(conv5)
    conv5 = layers.Conv2DTranspose(1, 1, padding='same')(conv5) #\bar{#dot{\omega}} Fig.4
    conv5 = layers.Activation('tanh')(conv5)

    model = Model(inputs=inputs, outputs=conv5)

    return model
