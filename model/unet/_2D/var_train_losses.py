import tensorflow as tf
import numpy as np

# Losses have to be defined as two function to enable loss weight tuning with rayTune.
# Inner functino only takes y_true and y_pred as input, outer functin defines additional parameters.
# Keras expects a custom loss to only have y_true and y_pred arguments


# Used for H2O prediction
# Weights before tuning: (1, 30, 200, 4e-1)
def custom_h2o_loss_with_weights(weights=(1.2, 2.9, 33, 0.1), cast=False):
    def custom_loss(y_true, y_pred):
        # Weights for different types of losses
        w_l1, w_l2, w_grad, w_ssim = weights
        #print(tf.shape(y_true))
        #print(tf.shape(y_pred))

        if cast:
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            y_pred = tf.squeeze(y_pred, axis=-1)

        l1_loss = tf.reduce_mean(tf.abs(tf.math.subtract(y_true,y_pred)))

        l2_loss = tf.reduce_mean(tf.math.squared_difference(y_true,y_pred))

        grad_loss = tf.reduce_mean(tf.math.squared_difference(tf.convert_to_tensor(np.gradient(y_true.numpy(), axis=2, edge_order=2)),
                                                                     tf.convert_to_tensor(np.gradient(y_pred.numpy(), axis=2, edge_order=2)))) + \
                    tf.reduce_mean(tf.math.squared_difference(tf.convert_to_tensor(np.gradient(y_true.numpy(), axis=1, edge_order=2)),
                                                                     tf.convert_to_tensor(np.gradient(y_pred.numpy(), axis=1, edge_order=2))))

        #print(tf.shape(y_true))
        #print(tf.shape(y_pred))
        ssim_loss = 1-tf.reduce_mean(tf.image.ssim(tf.expand_dims(y_true, axis=-1),tf.expand_dims(y_pred, axis=-1),max_val=2))

        total_loss = w_l1*l1_loss + w_l2*l2_loss + w_grad*grad_loss + w_ssim*ssim_loss  # weighted sum of different losses
        #print(f"Absolute l1 loss: {l1_loss}")
        #print(f"Absolute l2 loss: {l2_loss}")
        #print(f"Absolute grad loss: {grad_loss}")
        #print(f"Absolute ssim loss: {ssim_loss}")
        return total_loss
    return custom_loss


# Used for RHO prediction
def custom_rho_loss_with_weights(weights=(0.88994, 0.06), cast=False):
    def custom_loss(y_true, y_pred):

        # Weights for combining different losses
        w_pixel, w_grad = weights

        # Calculate the Pixel loss and Gradient loss
        if cast:
            y_true = tf.cast(y_true, tf.float64)
            y_pred = tf.cast(y_pred, tf.float64)
            y_pred = tf.squeeze(y_pred, axis=-1)
        pixel_loss = tf.reduce_mean(tf.abs(tf.math.subtract(y_true, y_pred)))
        grad_loss = tf.reduce_mean(tf.math.squared_difference(tf.convert_to_tensor(np.gradient(y_true.numpy(), axis=2, edge_order=2)),
                                                             tf.convert_to_tensor(np.gradient(y_pred.numpy(), axis=2, edge_order=2)))) + \
                    tf.reduce_mean(tf.math.squared_difference(tf.convert_to_tensor(np.gradient(y_true.numpy(), axis=1, edge_order=2)),
                                                              tf.convert_to_tensor(np.gradient(y_pred.numpy(), axis=1, edge_order=2))))
        # print(f"Absolute pixel loss: {pixel_loss}, weight: {w_pixel}")
        # print(f"Absolute grad loss: {grad_loss}, weight: {w_grad}")
        pixel_loss_val = pixel_loss * w_pixel
        grad_loss_val = grad_loss * w_grad
        total_loss = pixel_loss_val + grad_loss_val

        return total_loss
    return custom_loss