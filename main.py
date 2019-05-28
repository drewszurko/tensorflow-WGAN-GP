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

import os

from absl import app
from absl import flags
from tensorflow import keras
from tensorflow.python.ops import control_flow_util

import models

keras.backend.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 50, 'Epochs to train.')
flags.DEFINE_integer('batch_size', 64, 'Size of image batch.')
flags.DEFINE_integer('image_size', 64, 'I/O Image size.')
flags.DEFINE_integer('z_size', 128, 'Random vector noise size.')
flags.DEFINE_float('g_lr', .0001, 'Generator learning rate.')
flags.DEFINE_float('d_lr', .0001, 'Discriminator learning rate.')
flags.DEFINE_enum(
    'dataset', None,
    ['cifar10', 'celeb_a', 'tf_flowers', 'oxford_flowers102', 'oxford_iiit_pet'],
    'Dataset to train.')
flags.DEFINE_boolean('crop', False, 'Center crop images.')
flags.DEFINE_string('output_dir', '.', 'Output directory.')
flags.DEFINE_float('g_penalty', 10.0, 'Gradient penalty weight.')
flags.DEFINE_integer('n_critic', 5, 'Critic updates per generator update.')
flags.DEFINE_integer('n_samples', 64, 'Number of samples to generate.')
flags.mark_flag_as_required('dataset')


def main(argv):
    del argv

    pipeline = models.DatasetPipeline()
    dataset = pipeline.load_dataset()

    wgangp = models.WGANGP(dataset_info=pipeline.dataset_info)
    wgangp.train(dataset=dataset)


if __name__ == '__main__':
    app.run(main)
