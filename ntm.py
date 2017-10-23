import theano
import theano.tensor as T

import numpy as np

from six.moves import cPickle

from heads import writeHead, readHead
from utils import softplus, sigmoid, init_weight, focus_shift, circular_convolution, pass_or_fail, RMSprop 

from memory import mem_read, mem_write, mem_focus

# * indicates that a dimension is broadcastable

SMALL_CONSTANT = 1e-8

class NTM:
    def __init__(self, controller, output_size,
                 memory_slots = 32, slot_size = 10,
                 read_heads = 1,
                 batch_size = 10):
        """
            NTM.__init__(controller, memory_slots, slot_size, read_heads, batch_size) -> None

            initializes a Neural Turing Machine

            @param controller: controller is another class, which must have a method called `forward` and another called `get_weights`. This controller will process the read_vectors and the external input. This implemenation modularizes the NTM, meaning that the controller can be defined however the user wants to.
            @param memory_slots: the value M, the number of memory slots we have available
            @param slot_size: the value N, how much data can be stored in each individual slot
            @param read_heads: the number of read heads we have
            @param batch_size: batch size
        """

        self.controller = controller
        self.output_size = output_size

        # This represents the OUTPUT size of the controller
        self.controller_size = controller.size

        self.memory_slots = memory_slots 
        self.slot_size = slot_size

        self.batch_size = batch_size

        self.read_heads = [readHead(controller_size = self.controller_size,
                                    memory_slots = self.memory_slots,
                                    slot_size = self.slot_size,
                                    batch_size = self.batch_size
                                    ) for i in range (read_heads)]
        self.write_head = writeHead(controller_size = self.controller_size,
                                     memory_slots = self.memory_slots,
                                     slot_size = self.slot_size,
                                     batch_size = self.batch_size)
    
        self.output_weight = init_weight(self.controller_size, self.output_size)

        self.weights = [self.output_weight]
        #self.weights = []
        for head in self.read_heads:
            self.weights += head.get_weights()
        
        self.weights += self.write_head.get_weights()
        self.weights += self.controller.get_weights()
    
    # sequences, prior results, non_sequences
    def process_step(self, external_input, memory, read_vectors, previous_weights):
        """ 
            NTM.process(external_input, read_vector, previous_weights) -> new_memory, new_read_vectors, new_weights, output

            the main method of this entire program. Processes the data with a NTM.

            @param external_input: a batchsize x input_size matrix that represents the data for this timestep
            @param memory: a batchsize x N x M memory 3-tensor from the previous timestep
            @param read_vector: a self.read_heads x batchsize x N 3-tensor that represents all of the read vectors produced by read heads
            @param previous_weights: a 1 + self.read_heads x batchsize x M 3-tensor that represents all of the weightings produced by all heads in the previous timestep
        """

        # concatenated_read_vector -> batchsize x self.slot_size * read_vector
        concatenated_read_vector = T.concatenate([read_vectors[head] for head in range (len(self.read_heads))], axis = 1)

        # concatenated_input -> batchsize x self.slot_size * read_vector + external_input_length
        concatenated_input = T.concatenate([concatenated_read_vector, external_input], axis = 1)

        # controller_output -> batchsize x self.controller_size
        controller_output = T.nnet.relu(self.controller.forward(concatenated_input))

        # Represents NTM's actual output for this timestep
        # ntm_output -> batchsize x self.output_size
        ntm_output = T.dot(controller_output, self.output_weight)

        # let's get reading out of the way first
        for head in range (len(self.read_heads)):
            key, shift, sharpen, strengthen, interpolation = self.read_heads[head].produce(controller_output)

            # preprocess the values in preparation for mem_focus
            # key -> batchsize x 1 x N
            # strengthen -> batchsize x 1 (already good)
            key = key.dimshuffle([0, 'x', 1])

            # Focus by content + strengthen
            # preliminary_weight -> batchsize x M
            preliminary_weight = mem_focus(memory, key, strengthen)

            # Focus by location 
            interpolated_weight = interpolation * preliminary_weight + (1 - interpolation) * previous_weights[head]

            # Shift
            # Both arguments are batchsize x M, first being weighting, second being shift

            shifted_weight = focus_shift(interpolated_weight, shift, self.memory_slots)

            # Sharpen
            # We added broadcasted the second axis of sharpen, remember? :))))))))))))))))))))))))))
            # sharpened_weight -> batchsize x M
            sharpened_weight = shifted_weight ** sharpen

            # Normalize
            # T.sum(...) -> batchsize x 1
            final_weight = sharpened_weight / (T.sum(sharpened_weight, axis = 1, keepdims = True) + SMALL_CONSTANT)

            # read, read, read!
            # read_vec -> batchsize x N x 1, so we gotta flatten to batchsize x N 
            read_vec = mem_read(memory, final_weight)
            read_vectors = T.set_subtensor(read_vectors[head], T.flatten(read_vec, outdim = 2))

            previous_weights = T.set_subtensor(previous_weights[head], final_weight)
        
        # let's write now!
        key, add, erase, shift, sharpen, strengthen, interpolation = self.write_head.produce(controller_output)

        # preprocess the values in preparation for mem_focus
        # key -> batchsize x 1 x N
        # strengthen -> batchsize x 1 (already_good)

        key = key.dimshuffle([0, 'x', 1])

        # Focus by content + strengthen
        # preliminary_weight -> batchsize x M
        preliminary_weight = mem_focus(memory, key, strengthen)

        # Focus by location 
        interpolated_weight = interpolation * preliminary_weight + (1 - interpolation) * previous_weights[-1]

        # Shift
        shifted_weight = focus_shift(interpolated_weight, shift, self.memory_slots)

        # Sharpen
        sharpened_weight = shifted_weight ** sharpen

        # Normalize
        # T.sum(...) -> batchsize x 1
        final_weight = sharpened_weight / (T.sum(sharpened_weight, axis = 1, keepdims = True) + SMALL_CONSTANT)

        previous_weights = T.set_subtensor(previous_weights[-1], final_weight)
        
        # preprocess the values in preparation for mem_write
        # weighting -> batchsize x 1 x M
        # erase_vector -> batchsize x N x 1
        # add_vector -> batchsize x N x 1
        final_weight = final_weight.dimshuffle([0, 'x', 1])
        erase = erase.dimshuffle([0, 1, 'x'])
        add = add.dimshuffle([0, 1, 'x'])

        new_memory = mem_write(memory, final_weight, erase, add)

        # phew, almost done!
        return new_memory, read_vectors, previous_weights, ntm_output
    

     #external_input, memory, read_vectors, previous_weights

    def process(self, data, rand):
        prev = np.zeros(shape = [len(self.read_heads) + 1, self.batch_size, self.memory_slots])
        for head in range (len(self.read_heads) + 1):
            prev[head, :, 0] = 1

        [memory_states, read_vector_states, previous_weight_states, output_states], updates = theano.scan(fn = self.process_step, sequences = data, outputs_info = [rand, T.zeros(shape = [len(self.read_heads), self.batch_size, self.slot_size]) + SMALL_CONSTANT, theano.shared(prev), None])

        return memory_states, read_vector_states, previous_weight_states, output_states


