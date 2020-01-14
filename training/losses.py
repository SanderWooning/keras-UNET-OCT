from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

def luisa_coef(y_true, y_pred, nlabels=3):
    accum_inter = 0
    accum_union = 0
    for i in range(nlabels):
        w = 1. / (K.sum(y_true[:, :, i]) ** 2)
        print(w)
        dice_intersection = K.sum(K.abs(y_true[:, :, i] * y_pred[:, :, i]))
        dice_union = K.sum(K.square(y_true[:, :, i])) + K.sum(K.square(y_pred[:, :, i]))
        accum_inter += w * dice_intersection
        accum_union += w * dice_union
    bk_true = tf.dtypes.cast((tf.where(y_true[:, :, 0] == 0, 1, 0) * tf.where(y_true[:, :, 1] == 0, 1, 0) * tf.where(
        y_true[:, :, 2] == 0, 1, 0)), tf.float32)
    bk_pred = tf.dtypes.cast((tf.where(y_pred[:, :, 0] == 0, 1, 0) * tf.where(y_pred[:, :, 1] == 0, 1, 0) * tf.where(
        y_pred[:, :, 2] == 0, 1, 0)), tf.float32)
    w = 1. / (K.sum(bk_true) ** 2)
    print(w)
    dice_intersection = K.sum(K.abs(bk_true * bk_pred))
    dice_union = K.sum(K.square(bk_true)) + K.sum(K.square(bk_pred))
    accum_inter += w * dice_intersection
    accum_union += w * dice_union
    return (2 * accum_inter / accum_union)


def luisa_loss(y_true, y_pred, nlabels=3):
    return 1 - luisa_coef(y_true, y_pred, nlabels)


def generalized_dice_coef_7(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0, 1, 2))
    w = 1 / (w ** 2 + 0.000001)
    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, (0, 1, 2, 3))
    numerator = K.sum(numerator)

    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, (0, 1, 2, 3))
    denominator = K.sum(denominator)

    gen_dice_coef = 2 * numerator / denominator

    return gen_dice_coef


def generalized_dice_loss_7(y_true, y_pred):
    return 1 - generalized_dice_coef_7(y_true, y_pred)


def generalised_dice_loss_3d(y_true, y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(y_true, axis=[1, 2, 3])
    w = 1 / (w ** 2 + smooth)

    numerator = y_true * y_pred
    numerator = w * tf.reduce_sum(numerator, axis=[1, 2, 3])
    numerator = tf.reduce_sum(numerator, axis=1)

    denominator = y_pred + y_true
    denominator = w * tf.reduce_sum(denominator, axis=[1, 2, 3])
    denominator = tf.reduce_sum(denominator, axis=1)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss


def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'float32'), num_classes=3)[..., 0:])
    y_pred_f = K.flatten(y_pred[..., 0:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_9cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_9cat(y_true, y_pred)


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)




def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


def generalised_dice_loss_2d(y_true, y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(y_true, axis=[1, 2])
    w = 1 / (w ** 2 + smooth)

    numerator = y_true * y_pred
    numerator = w * tf.reduce_sum(numerator, axis=[1, 2])
    numerator = tf.reduce_sum(numerator, axis=1)

    denominator = y_pred + y_true
    denominator = w * tf.reduce_sum(denominator, axis=[1, 2])
    denominator = tf.reduce_sum(denominator, axis=1)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
           (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.01), 'float32') * K.cast(K.less(averaged_mask, 0.99), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss


def weighted_log_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # weights are assigned in this order : normal,necrotic,edema,enhancing
    weights = np.array([1, 5, 2, 4])
    weights = K.variable(weights)
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

def gen_dice_loss(y_true, y_pred):
    generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights
    return GDL+weighted_log_loss(y_true,y_pred)