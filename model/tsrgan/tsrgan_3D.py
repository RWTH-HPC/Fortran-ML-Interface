import os
import tensorflow as tf
# for TensorFlow 2.16.1 and higher: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
import tf_keras as keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from keras.layers import UpSampling3D, Add, Conv3D, Dense, Input, LeakyReLU, Concatenate
from keras.layers import PReLU, BatchNormalization, Lambda, Dropout, Activation
from keras.models import Model


def lrelu1(x):
    return tf.maximum(x, 0.25 * x)


def lrelu2(x):
    return tf.maximum(x, 0.3 * x)


def dense_block(input):
    """Residual Dense Block"""
    x1 = Conv3D(64, kernel_size=3, strides=1, padding='same', use_bias=True)(input)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([input, x1])

    x2 = Conv3D(64, kernel_size=3, strides=1, padding='same', use_bias=True)(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([input, x1, x2])

    x3 = Conv3D(64, kernel_size=3, strides=1, padding='same', use_bias=True)(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([input, x1, x2, x3])

    x4 = Conv3D(64, kernel_size=3, strides=1, padding='same', use_bias=True)(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([input, x1, x2, x3, x4])  # added x3, which ESRGAN didn't include

    x5 = Conv3D(64, kernel_size=3, strides=1, padding='same', use_bias=True)(x4)
    x5 = Lambda(lambda x: x * 0.2)(x5)
    """here: assumed beta=0.2"""
    x = Add()([x5, input])
    return x


def RRDB(input):
    """Residual in Residual Dense Block"""
    x = dense_block(input)
    x = dense_block(x)
    x = dense_block(x)
    """here: assumed beta=0.2 as well"""
    x = Lambda(lambda x: x * 0.2)(x)
    out = Add()([x, input])
    return out


def upsample(x_in, num_filters=64):
    """Definition of the last two blocks of the generator network
    # x_in: Input shape
    # int num_filters: number of filters
    # :return: model of the last two blocks"""
    #x = Conv3D(filters=num_filters, kernel_size=3, strides=1, padding='same')(x_in)
    # x = SubpixelConv3D('upSampleSubPixel_' + str(number), 2)(x)  # PixelShuffler x
    x = UpSampling3D(size=2)(x_in)
    x = Conv3D(filters=num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    return x

# def SubpixelConv3D(name, scale=2):
#    """
# Keras layer to do subpixel convolution.
# NOTE: Tensorflow backend only. Uses tf.depth_to_space
# :param scale: upsampling scale compared to input_shape. Default=2
# :return:
# """

# def subpixel_shape(input_shape):
#    dims = [input_shape[0],
#            None if input_shape[1] is None else input_shape[1] * scale,
#            None if input_shape[2] is None else input_shape[2] * scale,
#            int(input_shape[3] / (scale ** 2))]
#    output_shape = tuple(dims)
#    return output_shape

# def subpixel(x):
#    return tf.nn.depth_to_space(x, scale)
# return Lambda(subpixel, output_shape=subpixel_shape, name=name)

##############################
##      TSRGAN GENERATOR    ##
##############################

def generator_tsrgan3D(num_filters=64, channels=3, upfactor=4):
    """
    Definition of the generator
    int num_filters: number of filters
    int channels: number of channels
    :return: generator model
    """
    # if upscaling_factor not in [1, 2, 4, 8, 16, 32, 64, 128]:
    #    raise ValueError('Upscaling factor must be either 1, 2, 4, or 8. You chose {}'.format(upscaling_factor))

    lr_inputs = Input(shape=(None, None, None, channels))  # [0, 1]
    #lr_inputs = Input(shape=(33, 33, 33, channels))  # [0, 1]

    # Pre-residual
    x_start = Conv3D(filters=num_filters, data_format="channels_last", kernel_size=3, strides=1, padding='same', name="conv3D_1")(lr_inputs)
    x_start = LeakyReLU(0.2)(x_start)

    # Residual-in-Residual Dense Block (3 Dense Block by default)
    x = RRDB(x_start)

    # Post-residual block
    x = Conv3D(filters=num_filters, kernel_size=3, strides=1, padding='same', name="conv3D_afterRRDB")(x)
    x = Lambda(lambda x: x * 0.2)(x)
    x = Add()([x, x_start])

    x = upsample(x, num_filters=64)
    if upfactor==4:
        x = upsample(x, num_filters=64)

    # Mr. Bode version
    #x = Conv3D(filters=num_filters * 8, kernel_size=3, strides=1, padding='same', activation=lrelu1)(x)
    #x = Conv3D(filters=num_filters * 8, kernel_size=3, strides=1, padding='same', activation=lrelu1)(x)

    # Final 2 convolutional layers
    x = Conv3D(filters=num_filters, kernel_size=3, strides=1, padding='same', name="conv3D_Final")(x)
    x = LeakyReLU(0.2)(x)

    x = Conv3D(filters=channels, kernel_size=9, strides=1, padding='same')(x)  # [-inf, -inf] -> [-1,1] (tahn) #Kernal size 3 or 9
    hr_output = Activation(activation='tanh', dtype='float32')(x)
    #hr_output = Conv3D(filters=channels, kernel_size=3, strides=1, padding='same')(x)  # [-inf, -inf] -> [-1,1] (tahn) #Kernal size 3 or 9

    return Model(lr_inputs, hr_output, name="generator_tsrgan3D")


##################################
##      TSRGAN DISCRIMINATOR     #
##################################

def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, name=None):
    """Definition of the discriminator block
    x_in: input shape
    int num_filters: number of filters
    int strides: number of strides
    bool batchnorm: enable batchnormalization block
    string name: name of the layer
    :return: discriminator block
    """
    # x = Conv3D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
    # kernel_initializer=_kernel_init(), kernel_regularizer=_regularizer(0.), name=name)(x_in)
    x = Conv3D(filters=num_filters, kernel_size=3, strides=strides, padding='same', name=name)(x_in)
    x = LeakyReLU(alpha=0.2)(x)
    if batchnorm:
        #x = BatchNormalization(momentum=0.8, fused=True)(x)  # Use faster fused implementation
        x = BatchNormalization(momentum=0.8)(x)

    return x


def discriminator_tsrgan3D(num_filters=64, channels=3):  # PIESRGAN Version
    """
    Definition of the discriminator
    int num_filters: number of filters
    int channels: number of channels
    :return: discriminator model
    """

    img = Input(shape=(None, None, None, channels))  # [0, 1]

    x = discriminator_block(img, num_filters, batchnorm=False, name='conv0_0')
    x = discriminator_block(x, num_filters, strides=2, name='conv0_1')

    x = discriminator_block(x, num_filters * 2, name='conv1_0')
    x = discriminator_block(x, num_filters * 2, strides=2, name='conv1_1')

    x = discriminator_block(x, num_filters * 4, name='conv2_0')
    x = discriminator_block(x, num_filters * 4, strides=2, name='conv2_1')

    x = discriminator_block(x, num_filters * 8, name='conv3_0')
    x = discriminator_block(x, num_filters * 8, strides=2, name='conv3_1')

    x = Dense(num_filters * 16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    out = Dense(units=1, name='linear2', dtype='float32')(x)  # [0, 1] -> [0, 1] 0: fake    1: real

    return Model(img, out, name="discriminator_tsrgan3D")
