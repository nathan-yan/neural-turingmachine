import theano
import theano.tensor as T

import numpy as np
from matplotlib import pyplot as plt

# * indicates that a dimension is broadcastable

class NTM:
    def __init__(self, controller_size = 100,
                 memory_slots = 32, slot_size = 10,
                 read_heads = 1,
                 batch_size = 10):
        self.controller_size = controller_size

        self.memory_slots = memory_slots 
        self.slot_size = slot_size

        self.batch_size = batch_size

        self.read_heads = [readHead(controller_size = self.controller_size,
                                    memory_slots = self.memory_slots,
                                    slot_size = self.slot_size,
                                    batch_size = self.batch_size
                                    ) for i in range (read_heads)]
        self.write_heads = writeHead(controller_size = self.controller_size,
                                     memory_slots = self.memory_slots,
                                     slot_size = self.slot_size,
                                     batch_size = self.batch_size)
    
    # sequences, prior results, non_sequences
    def process(self, data, ):
        """ 
            process()
        """
        pass
    

class readHead:
    def __init__(self, controller_size,
                 memory_slots, slot_size, batch_size):
        self.controller_size = controller_size
        self.memory_slots = memory_slots
        self.slot_size = slot_size

        self.batch_size = batch_size

        # For now we'll only allow one shift backwards or forwards
        self.weights = {
                            "controller->key"           : init_weight(self.controller_size, self.slot_size),
                            "controller->shift"         : init_weight(self.controller_size, 3),     
                            "controller->sharpen"       : init_weight(self.controller_size, 1),        
                            "controller->strengthen"    : init_weight(self.controller_size, 1),
                            "controller->interpolation" : init_weight(self.controller_size, 1),
                       }
        
    def produce(self, controller):
        """
            readHead.recurrence(controller, previous_weight) -> key           (batchsize x N),
                                                                shift         (batchsize x 3),
                                                                sharpen       (batchsize x 1),
                                                                strengthen    (batchsize x 1),
                                                                interpolation (batchsize x 1)
            
            produces controller parameters to manipulate/read memory

            @param controller: a batchsize x controller_size matrix, representing the output of the controller
        """

        # key, add, erase -> batchsize x N
        key = T.dot(controller, self.weights["controller->key"])
        
        # shift -> batchsize x 3 
        shift = T.nnet.softmax(T.dot(controller, self.weights["controller->shift"]))        # SOFTMAX

        backward_shift = shift[:, 0]
        stay_forward_shift = shift[:, 1:3]      # represents the shift values for STAY and FORWARD

        zeros_size = self.memory_slots - 3

        # We are concatenating along the second axis, we're basically moving the first element (which represents the backward shift) to the front
        # ex:
        # There are 7 memory slots 
        # Useless zeros are wrapped in [] to increase history.
        # 0.2 0.9 0.1 [0.0 0.0 0.0 0.0] -> 0.9 0.1 [0.0 0.0 0.0 0.0] 0.2
        true_shift = T.concatenate([stay_forward_shift, T.zeros([self.batch_size, zeros_size]), backward_shift.reshape([self.batch_size, 1])], axis = 1)      # WRAP

        # sharpen, strengthen, interpolation -> batchsize x 1
        # sharpen and strengthen must both be greater than or equal to 1, so we'll apply the softplus function (Graves et al., 2016)
        sharpen = softplus(T.dot(controller, self.weights["controller->sharpen"]))      # SOFTPLUS                         
        strengthen = softplus(T.dot(controller, self.weights["controller->strengthen"]))        # SOFTPLUS 

        interpolation = T.nnet.sigmoid(T.dot(controller, self.weights["controller->interpolation"]))        # SIGMOID

        return key, true_shift, sharpen, strengthen, interpolation

class writeHead:
    def __init__(self, controller_size,
                 memory_slots, slot_size,
                 batch_size):
        self.controller_size = controller_size
        self.memory_slots = memory_slots
        self.slot_size = slot_size

        self.batch_size = batch_size

        # For now we'll only allow one shift backwards or forwards
        self.weights = {
                            "controller->key"           : init_weight(self.controller_size, self.slot_size),
                            "controller->add"           : init_weight(self.controller_size, self.slot_size),
                            "controller->erase"         : init_weight(self.controller_size, self.slot_size),
                            "controller->shift"         : init_weight(self.controller_size, 3),     
                            "controller->sharpen"       : init_weight(self.controller_size, 1),        
                            "controller->strengthen"    : init_weight(self.controller_size, 1),
                            "controller->interpolation" : init_weight(self.controller_size, 1),
                       }
    
    def produce(self, controller):
        """
            writeHead.recurrence(controller, previous_weight) -> key           (batchsize x N),
                                                                 add           (batchsize x N),
                                                                 erase         (batchsize x N),
                                                                 shift         (batchsize x 3),
                                                                 sharpen       (batchsize x 1),
                                                                 strengthen    (batchsize x 1),
                                                                 interpolation (batchsize x 1)
            
            produces controller parameters to manipulate/write memory

            @param controller: a batchsize x controller_size matrix, representing the output of the controller
        """

        # key, add, erase -> batchsize x N
        key = T.dot(controller, self.weights["controller->key"])
        add = T.dot(controller, self.weights["controller->add"])
        erase = T.nnet.sigmoid(T.dot(controller, self.weights["controller->erase"]))        # SIGMOID

        # shift -> batchsize x 3 
        shift = T.nnet.softmax(T.dot(controller, self.weights["controller->shift"]))        # SOFTMAX

        backward_shift = shift[:, 0]
        stay_forward_shift = shift[:, 1:3]      # represents the shift values for STAY and FORWARD

        zeros_size = self.memory_slots - 3

        # We are concatenating along the second axis, we're basically moving the first element (which represents the backward shift) to the front
        # ex:
        # There are 7 memory slots 
        # Useless zeros are wrapped in [] to increase history.
        # 0.2 0.9 0.1 [0.0 0.0 0.0 0.0] -> 0.9 0.1 [0.0 0.0 0.0 0.0] 0.2
        true_shift = T.concatenate([stay_forward_shift, T.zeros([self.batch_size, zeros_size]), backward_shift.reshape([self.batch_size, 1])], axis = 1)      # WRAP

        # sharpen, strengthen, interpolation -> batchsize x 1
        # sharpen and strengthen must both be greater than or equal to 1, so we'll apply the softplus function (Graves et al., 2016)
        sharpen = softplus(T.dot(controller, self.weights["controller->sharpen"]))      # SOFTPLUS                         
        strengthen = softplus(T.dot(controller, self.weights["controller->strengthen"]))        # SOFTPLUS 

        interpolation = T.nnet.sigmoid(T.dot(controller, self.weights["controller->interpolation"]))        # SIGMOID

        return key, add, erase, true_shift, sharpen, strengthen, interpolation

def softplus(x):
    """
        softplus(x) -> result (any shape)

        applies softplus to x

        @param x: the value we want to apply the softplus function to
    """

    result = 1 + T.log(1 + T.exp(x))
    return result

def init_weight(input_size, output_size):
    return theano.shared((np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))))

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

def circular_convolution(weight_idx, shift, weight):
    """
        circular_convolution(weight_idx, shift, weighting) -> column (batchsize)
        
        This function follows the circular convolution rule: column[i] += old_weight[j] * shift[(i - j) % memory_slots], for any particular i. `column` has a batchsize, so `column` will actually be a batchsize column vector. 

        @param weight_idx: a scalar that represents the column index this function is producing, or, in the above example, i
        @param shift: a batchsize x M matrix that represents the shift vectors
        @param weight: a batchsize x M matrix that represents the weight vectors 
    """

    def conv(shift_idx, current_value, shift, weight_idx, weight):
        current_value += weight[:, shift_idx] * shift[:, (weight_idx - shift_idx)%5]

        return current_value

    # memory_size -> 1, represents the size of weight -> batchsize x M in its last dimension (M)
    memory_size = weight.shape[-1]

    # batch_size -> 1, represents the size of weight -> batchsize x M in its first dimension (batchsize)
    batch_size = weight.shape[0]

    # columns -> timesteps x batchsize
    columns, updates = theano.scan(fn = conv, sequences = [T.arange(memory_size)], outputs_info = [T.zeros(shape = [batch_size])], non_sequences = [shift, weight_idx, weight])

    # column -> batchsize
    column = columns[-1]
    return column

def focus_shift(weight, shift):
    """
        focus_shift(weightings, shift) -> new_weighting

        `circular_convolution` produces the value of the new_weighting for a single memory position. focus_shift applies the circular_convolution to every memory slot using scan.

        @param weight: a batchsize x M matrix that represents the weight vectors
        @param shift: a batchsize x M matrix that represents the shift vectors
    """

    # memory_size -> 1, represents the size of weight -> batchsize x M in its last dimension (M)
    memory_size = weight.shape[-1]

    # new_weighting -> M x batchsize, but will be transposed to have the proper ordering
    new_weighting, updates = theano.scan(fn = circular_convolution, sequences = [T.arange(memory_size)], non_sequences = [shift, weight])

    return new_weighting.dimshuffle([1, 0])

def pass_or_fail(correct, answer):
    if (np.sum((correct - answer) ** 2) < 0.01):
        print("PASSED") 
    else:
        print("FAILED, squared loss is " + str(np.sum((correct - answer) ** 2)))

def test():
     # <mem_write and read TEST>

    # memory -> 2 x 4 x 3
    memory = np.array([[[1, 1, 1, 1], [1, 2, 1, 2],[8, 1, 1, 3]], [[20, 2, 3, 4], [1, 1, 2, 3], [2, 1, 10, 1]]]).transpose([0, 2, 1])

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

    w_W = mem_write(m, w, e, wr)
    w_R = mem_read(m, w.dimshuffle([0, 2, 1]))

    f_w = theano.function([m, w, e, wr], w_W)
    f_r = theano.function([m, w], w_R)

    print ("TEST mem_read", end = " "),
    correct = np.array([[[4.5], [1.5], [1], [2.5]], [[4.9], [2.1], [12.1], [4.1]]])
    answer = f_r(memory, weighting)

    pass_or_fail(correct, answer)

    print ("TEST mem_write", end = " "),
    correct = np.array([[[1, 1, 4.5], [1, 51.3, 50.65], [1, 1.75, 1.75], [1, 3, 4]], [[19, -8, -8], [2.14, 2.53, 2.7],[1.47, -11.68, -6], [3.65, 0.75, 0.5]]])
    answer = f_w(memory, weighting, erase, write)
    
    pass_or_fail(correct, answer)

    # </mem_write and read TEST> PASSED

    # <mem_focus TEST>
    print("TEST mem_focus", end = " "),
    
    # memory -> 2 x 4 x 3
    memory = np.array([[[1, 1, 1, 1], [10, 22, 1, 2],[80, 1, 1, 3]], [[20, 2, 3, 4], [1, 1, 2, 3], [2, 1, 10, 1]]]).transpose([0, 2, 1])
    # key -> 2 x 1 x 4 
    key = np.array([[1, 1, 1, 1], [1, 1, 1, 1]]).reshape([2, 1, 4])
    strength = np.array([[1], [2]])

    m = T.tensor3()
    k = T.tensor3()
    s = T.matrix()

    w_F = mem_focus(m, k, s)

    f = theano.function([m, k, s], w_F)
    correct = np.array([[0.41979, 0.3176181, 0.26258, ], [0.28875, 0.433907, 0.277338]])
    
    answer = f(memory, key, strength)

    pass_or_fail(correct, answer)

    # </mem_focus TEST> PASSED

    # <focus_shift TEST>
    print("TEST focus_shift", end = " "),
    w = T.matrix()
    s = T.matrix()             
    
    f = theano.function([w, s], [focus_shift(w, s)])

    weight = np.array([[0, 0, 0, 0, 1], [1, 2, 3, 4, 5]])
    shift = np.array([[.9, .1, 0, 0, 0], [0, 0, 0, 0, 1]])

    correct = np.array([[0.1, 0, 0, 0, 0.9], [2, 3, 4, 5, 1]])
    answer = f(weight, shift)[0]

    pass_or_fail(correct, answer)
    # </focus_shift TEST> PASSED

    # <writeHead TEST>
    print("TEST writeHead", end = " "),
    w = writeHead(controller_size = 10, memory_slots = 8, slot_size = 5, batch_size = 2)
    controller = np.random.randn(2, 10)

    c = T.matrix()

    values = w.produce(c)

    f = theano.function([c], values)
    answer = f(controller)

    assert(answer[0].shape == (2, 5))
    assert(answer[1].shape == (2, 5))
    assert(answer[2].shape == (2, 5))
    assert(answer[3].shape == (2, 8))
    assert(answer[4].shape == (2, 1))
    assert(answer[5].shape == (2, 1))
    assert(answer[6].shape == (2, 1))
    print("PASSED")
    # </writeHead TEST> PASSED

    # <readHead TEST>
    print("TEST readHead", end = " "),
    r = readHead(controller_size = 10, memory_slots = 8, slot_size = 5, batch_size = 2)
    controller = np.random.randn(2, 10)

    c = T.matrix()

    values = r.produce(c)

    f = theano.function([c], values)
    answer = f(controller)

    print (answer)

def main():
    pass

test()


