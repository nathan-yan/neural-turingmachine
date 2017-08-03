import theano
import theano.tensor as T

import numpy as np

from utils import softplus, sigmoid, init_weight, focus_shift, circular_convolution, pass_or_fail, RMSprop

SMALL_CONSTANT = 1e-8

def mem_write(memory, weighting, erase_vector, add_vector):
    """
        mem_write(memory, erase_vector, add_vector) -> written_memory (batchsize x N x M)

        writes to memory

        @param memory: a batchsize x N x M 3-tensor, where N is the size of a memory slot, and M is the number of memory slots
        @param weighting: a batchsize x 1 x M 3-tensor, that dictates the strength of the erase and add 
        @param erase_vector: a batchsize x N x 1 3-tensor (or technically matrix), that decides what to erase from every memory slot. A one is full erasure
        @param add_vector: a batchsize x N x 1 3-tensor, that decides what to add to every memory slot
    """

    # erase -> batchsize x N x M
    erase = 1 - T.batched_dot(erase_vector, weighting)                                                  

    # write -> batchsize x N x M
    write = T.batched_dot(add_vector, weighting)

    written_memory = memory * erase + write 
    return written_memory

def mem_read(memory, weighting):
    """
        mem_read(memory, weighting) -> read_vector (batchsize x N x 1)

        reads from memory

        @param memory: a batchsize x N x M 3-tensor, where N is the size of a memory slot, and M is the number of memory slots
        @param weighting: a batchsize x M x 1 3-tensor, how much we want to read from each memory position. Since weighting is softmaxed, we're taking a weighted average of the memory
    """    

    # read_vector -> batchsize x N x 1
    read_vector = T.batched_dot(memory, weighting)

    return read_vector

def mem_focus(memory, key, strength):
    """
        mem_focus(memory, key, strength) -> weighting (batchsize x M)

        produces a weighting over memory positions based on a key

        @param memory: a batchsize x N x M 3-tensor
        @param key: a batchsize x 1 x N 3-tensor. mem_focus() is expected to output a weighting for each batch element
        @param strength: a batchsize x 1 matrix, sharpens the weighting 
    """

    # dot -> batchsize x 1 x M
    dot = T.batched_dot(key, memory)

    # memory_magnitude -> batchsize x M 
    memory_magnitude = T.sqrt(T.sum(memory ** 2, axis = 1))

    # key_magnitude -> batchsize x 1*
    key_magnitude = T.addbroadcast(T.sqrt(T.sum(key ** 2, axis = 2)), 1)

    # multiplied_magnitude -> batchsize x 1 x M
    multiplied_magnitude = (memory_magnitude * key_magnitude).dimshuffle([0, 'x', 1])

    # cosine_similarity -> batchsize x 1 x M
    cosine_similarity = dot/(multiplied_magnitude + SMALL_CONSTANT)

    # strengthened_cosine_similarity -> batchsize x 1 x M
    strengthened_cosine_similarity = cosine_similarity * strength.dimshuffle([0, 1, 'x']) 

    # weighting -> batchsize x M
    weighting = T.nnet.softmax(T.flatten(strengthened_cosine_similarity, outdim = 2))

    return weighting

