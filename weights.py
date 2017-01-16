import theano
import theano.tensor as T 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

def init_weights(shape1, shape2):
    return theano.shared(np.asarray(np.random.randn(shape1, shape2) / np.sqrt(shape1), dtype = theano.config.floatX))

def init_weights_conv(shape1, shape2, shape3, shape4):
    return theano.shared(np.asarray(np.random.randn(shape1, shape2, shape3, shape4) / np.sqrt(shape2 * shape3 * shape4), dtype = theano.config.floatX))