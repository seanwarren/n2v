from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, move_channel_for_backend, axes_dict, axes_check_and_normalize, backend_channels_last
from ..internals.losses import loss_laplace, loss_mse, loss_mae, loss_thresh_weighted_decay, loss_noise2void, \
    loss_noise2voidAbs

import numpy as np


import keras.backend as K
from keras.callbacks import Callback, TerminateOnNaN
from keras.utils import Sequence


class ParameterDecayCallback(Callback):
    """ TODO """
    def __init__(self, parameter, decay, name=None, verbose=0):
        self.parameter = parameter
        self.decay = decay
        self.name = name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        old_val = K.get_value(self.parameter)
        if self.name:
            logs = logs or {}
            logs[self.name] = old_val
        new_val = old_val * (1. / (1. + self.decay * (epoch + 1)))
        K.set_value(self.parameter, new_val)
        if self.verbose:
            print("\n[ParameterDecayCallback] new %s: %s\n" % (self.name if self.name else 'parameter', new_val))


def prepare_model(model, optimizer, loss, training_scheme='CARE', metrics=('mse','mae'),
                  loss_bg_thresh=0, loss_bg_decay=0.06, Y=None):
    """ TODO """

    from keras.optimizers import Optimizer
    isinstance(optimizer,Optimizer) or _raise(ValueError())

    if training_scheme == 'CARE':
        loss_standard   = eval('loss_%s()'%loss)
    elif training_scheme == 'Noise2Void':
        if loss == 'mse':
            loss_standard = eval('loss_noise2void()')
        elif loss == 'mae':
            loss_standard = eval('loss_noise2voidAbs()')

    _metrics = [eval('loss_%s()' % m) for m in metrics]
    callbacks       = [TerminateOnNaN()]

    # checks
    assert 0 <= loss_bg_thresh <= 1
    assert loss_bg_thresh == 0 or Y is not None
    if loss == 'laplace':
        assert K.image_data_format() == "channels_last", "TODO"
        assert model.output.shape.as_list()[-1] >= 2 and model.output.shape.as_list()[-1] % 2 == 0

    # loss
    if loss_bg_thresh == 0:
        _loss = loss_standard
    else:
        freq = np.mean(Y > loss_bg_thresh)
        # print("class frequency:", freq)
        alpha = K.variable(1.0)
        loss_per_pixel = eval('loss_{loss}(mean=False)'.format(loss=loss))
        _loss = loss_thresh_weighted_decay(loss_per_pixel, loss_bg_thresh,
                                           0.5 / (0.1 + (1 - freq)),
                                           0.5 / (0.1 +      freq),
                                           alpha)
        callbacks.append(ParameterDecayCallback(alpha, loss_bg_decay, name='alpha'))
        if not loss in metrics:
            _metrics.append(loss_standard)


    # compile model
    model.compile(optimizer=optimizer, loss=_loss, metrics=_metrics)

    return callbacks


class DataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        return self.X[idx], self.Y[idx]


class Noise2VoidDataWrapper(Sequence):

    def __init__(self, X, Y, batch_size, num_pix=1, shape=(64, 64),
                 value_manipulation=None):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape = shape
        self.value_manipulation = value_manipulation
        self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape)
        self.dims = len(shape)
        self.n_chan = X.shape[-1]

        if self.dims < 2 or self.dims > 3:
            raise Exception('Dimensionality not supported.')

        self.X_Batches = np.zeros((X.shape[0],) + shape + (X.shape[-1],))
        self.Y_Batches = np.zeros((X.shape[0],) + shape + (Y.shape[-1],))

        self.box_size = self.get_box_size(shape, num_pix)
        self.rand_float = self.__rand_float_coordsND__(self.box_size, self.dims)


    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]
        
        self.patch_sampler(self.X, self.Y, self.X_Batches, self.Y_Batches, idx, self.range, self.shape)

        for j in idx:
            coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                shape=np.array(self.X_Batches.shape)[1:-1])

            y_val = []
            x_val = []
            for k in range(len(coords)):
                y_val.append(np.copy(self.Y_Batches[(j, *coords[k], ...)]))
                x_val.append(self.value_manipulation(self.X_Batches[j, ...], coords[k], self.dims))

            self.Y_Batches[j] *= 0

            for k in range(len(coords)):
                for c in range(self.n_chan):
                    self.Y_Batches[(j, *coords[k], c)] = y_val[k][c]
                    self.Y_Batches[(j, *coords[k], self.n_chan+c)] = 1
                    self.X_Batches[(j, *coords[k], c)] = x_val[k][c]


        return self.X_Batches[idx], self.Y_Batches[idx]

    @staticmethod
    def patch_sampler(X, Y, X_Batches, Y_Batches, indices, range, shape):
        for j in indices:
            start = [np.random.randint(0, r + 1) for r in range]
            coords = [slice(s,s+sh) for s,sh in zip(start,shape)]
            X_Batches[j] = X[(j, *coords)]
            Y_Batches[j] = Y[(j, *coords)]


    @staticmethod
    def get_stratified_coords(coord_gen, box_size, shape):
        coords = []
        box_count = [int(np.ceil(s / box_size)) for s in shape]
        for index in np.ndindex(tuple(box_count)):
            offset = next(coord_gen)
            index = tuple(int(i*box_size + o) for i,o in zip(index,offset))
            if all([i < s for i,s in zip(index,shape)]):
                coords.append(index)
        return coords

    @staticmethod
    def __rand_float_coordsND__(boxsize, dims):
        while True:
            yield tuple(np.random.rand(dims) * boxsize)

    @staticmethod
    def get_box_size(shape, num_pix):
        return np.round(np.power(np.product(shape) / num_pix, 1.0/len(shape))).astype(np.int)

