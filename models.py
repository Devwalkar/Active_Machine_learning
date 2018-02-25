"""The model definitions."""
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Flatten, merge
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K


def Convolution(f, k=3, s=2, border_mode='same', l2=.0, **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f, k, k, border_mode=border_mode, subsample=(s, s),
                         W_regularizer=WeightRegularizer(l2=l2), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(mode=mode, axis=axis, **kwargs)


def ConvBlock(i, nf, k=3, s=1, border_mode='same', drop_p=.0, l2=1e-2,
              maxpool=True, norm=False, **kwargs):
    """A Conv-BatchNorm-LeakyRelu block followed by Dropout."""
    x = Convolution(nf, k=k, s=s, border_mode=border_mode, l2=l2, **kwargs)(i)
    if maxpool:
        x = MaxPooling2D((2, 2))(x)

    x = LeakyReLU(0.2)(x)

    if norm:
        x = BatchNorm()(x)
    if drop_p > .0:
        x = Dropout(drop_p)(x)

    return x


def lesion_detector_parameterized(nf, l2=.0, input_size=512, n_blocks=4,
                                  drop_p=.0, context_blocks=0, lr=2e-4):
    """A model that outputs the likelihood of patches containing lesions."""
    img = Input(shape=(3,) + (input_size, input_size))
    out_size = int(input_size / (2**n_blocks))

    ###########################################################################
    #                           MAIN NETWORK DEFINITION                       #
    ###########################################################################
    xi = img
    nfi = nf
    for i in range(n_blocks):
        nfi = nf * 2**i
        if nfi > nf * 8:
            nfi = nf * 8

        xi = ConvBlock(xi, nfi, l2=l2, maxpool=True)
        xi = ConvBlock(xi, nfi, k=1, s=1, maxpool=False, l2=l2, drop_p=drop_p)

    xi = ConvBlock(xi, nfi, k=1, s=1, maxpool=False, drop_p=drop_p, l2=l2)
    xi = ConvBlock(xi, nfi, k=1, s=1, maxpool=False, drop_p=drop_p, l2=l2)

    ###########################################################################
    #                         CONTEXT NETWORK DEFINITION                      #
    ###########################################################################
    # Lesion classifier
    L = Convolution(1, k=1, s=1, activation='sigmoid')

    for _ in range(context_blocks):
        # Run the context network on top of previous layer
        c_out = ConvBlock(xi, nfi, l2=l2, maxpool=False)
        c_out = ConvBlock(c_out, nfi, k=1, s=1, l2=l2, drop_p=drop_p, maxpool=False)

        # Residual connection. Makes it possible to learn the identity function.
        xi = merge([xi, c_out], mode='sum')

    ###########################################################################
    #                           FINAL CLASSIFICATIONS                         #
    ###########################################################################
    lesions = L(xi)

    out = MaxPooling2D((out_size, out_size))(lesions)
    out = Flatten()(out)

    lesions_model = Model(img, lesions)
    model = Model(img, out)

    ###########################################################################
    #                               LOSS FUNCTIONS                            #
    ###########################################################################
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def acc(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        return K.mean(K.equal(y_true_flat, K.round(y_pred_flat)))

    def lesion_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L = K.binary_crossentropy(y_pred_flat, y_true_flat)
        L_balance = 5 * y_true_flat * L + (1 - y_true_flat) * L

        return L_balance

    lesions_model.compile(optimizer=opt, loss=lesion_loss, metrics=[acc])

    return model, lesions_model
