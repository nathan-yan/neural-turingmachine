import theano
import theano.tensor as T

import numpy as np
from matplotlib import pyplot as plt

# * indicates that a dimension is broadcastable

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
    cosine_similarity = dot/multiplied_magnitude

    # strengthened_cosine_similarity -> batchsize x 1 x M
    strengthened_cosine_similarity = cosine_similarity * strength.dimshuffle([0, 1, 'x'])

    # weighting -> batchsize x M
    weighting = T.nnet.softmax(T.flatten(strengthened_cosine_similarity, outdim = 2))

    return weighting

def test():
     # <mem_write and read TEST>
    # memory -> 2 x 4 x 3
    memory = np.array([[[1, 1, 1, 1], [1, 2, 1, 2],[8, 1, 1, 3]], [[20, 2, 3, 4], [1, 1, 2, 3], [2, 1, 10, 1]]]).transpose([0, 2, 1])
    print(memory)
    # weighting -> 2 x 1 x 3
    weighting = np.array([[0, .5, .5], [0.1, 0.9, 1]]).reshape([2, 1, 3])

    # erase -> 2, 4, 1
    erase = np.array([[1, 0.7, 0.5, 0], [0, 0.3, 0.1, 1]]).reshape([2, 4, 1])

    # write 
    write = np.array([[1, 100, 2, 2], [-10, 2, -15, .5]]).reshape([2, 4, 1])

    m = T.tensor3()
    w = T.tensor3()
    e = T.tensor3()
    wr = T.tensor3()

    #w_M = mem_write(m, w, e, wr)
    w_R = mem_read(m, w.dimshuffle([0, 2, 1]))

    #f = theano.function([m, w, e, wr], w_M)
    f = theano.function([m, w], w_R)

    #print(f(memory, weighting, erase, write))
    print(f(memory, weighting))

    # </mem_write and read TEST> PASSED

    # <mem_focus TEST>
    # memory -> 2 x 4 x 3
    memory = np.array([[[1, 1, 1, 1], [10, 22, 1, 2],[8, 1, 1, 3]], [[20, 2, 3, 4], [1, 1, 2, 3], [2, 1, 10, 1]]]).transpose([0, 2, 1])
    print(memory)
    # key -> 2 x 1 x 4 
    key = np.array([[1, 1, 1, 1], [1, 1, 1, 1]]).reshape([2, 1, 4])
    strength = np.array([[1], [2]])

    m = T.tensor3()
    k = T.tensor3()
    s = T.matrix()

    w_F = mem_focus(m, k, s)

    f = theano.function([m, k, s], w_F)
    print(f(memory, key, strength))
    # </mem_focus TEST> PASSED

def main():
    pass 

test()


