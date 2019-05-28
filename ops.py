# MIT License
#
# Copyright (c) 2019 Drew Szurko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow import optimizers
from tensorflow import reduce_mean
from tensorflow.python.keras import layers


class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(Conv2D, self).__init__()
        self.conv_op = layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=False,
                                     kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        return self.conv_op(inputs)


class UpConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(UpConv2D, self).__init__()
        self.up_conv_op = layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 use_bias=False,
                                                 kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        x = self.up_conv_op(inputs)
        return x


class BatchNorm(layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1, momentum=0.99):
        super(BatchNorm, self).__init__()
        self.batch_norm = layers.BatchNormalization(epsilon=epsilon,
                                                    axis=axis,
                                                    momentum=momentum)

    def call(self, inputs, **kwargs):
        x = self.batch_norm(inputs)
        return x


class LayerNorm(layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1):
        super(LayerNorm, self).__init__()
        self.layer_norm = layers.LayerNormalization(epsilon=epsilon, axis=axis)

    def call(self, inputs, **kwargs):
        x = self.layer_norm(inputs)
        return x


class LeakyRelu(layers.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, **kwargs):
        x = self.leaky_relu(inputs)
        return x


class AdamOptWrapper(optimizers.Adam):
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.,
                 beta_2=0.9,
                 epsilon=1e-4,
                 amsgrad=False,
                 **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate, beta_1, beta_2, epsilon,
                                             amsgrad, **kwargs)


def d_loss_fn(f_logit, r_logit):
    f_loss = reduce_mean(f_logit)
    r_loss = reduce_mean(r_logit)
    return f_loss - r_loss


def g_loss_fn(f_logit):
    f_loss = -reduce_mean(f_logit)
    return f_loss
