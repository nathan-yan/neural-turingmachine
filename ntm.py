import theano
import theano.tensor as T

import numpy as np
from matplotlib import pyplot as plt

# * indicates that a dimension is broadcastable

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

            shifted_weight = focus_shift(interpolated_weight, shift)

            # Sharpen
            # We added broadcasted the second axis of sharpen, remember? :))))))))))))))))))))))))))
            # sharpened_weight -> batchsize x M
            sharpened_weight = shifted_weight ** sharpen

            # Normalize
            # T.sum(...) -> batchsize x 1
            final_weight = sharpened_weight / T.sum(sharpened_weight, axis = 1, keepdims = True)

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
        shifted_weight = focus_shift(interpolated_weight, shift)

        # Sharpen
        sharpened_weight = shifted_weight ** sharpen

        # Normalize
        # T.sum(...) -> batchsize x 1
        final_weight = sharpened_weight / T.sum(sharpened_weight, axis = 1, keepdims = True)

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

    def process(self, data):
        [memory_states, read_vector_states, previous_weight_states, output_states], updates = theano.scan(fn = self.process_step, sequences = data, outputs_info = [theano.shared(np.random.randn(self.batch_size, self.slot_size, self.memory_slots) * 0.05), T.zeros(shape = [len(self.read_heads), self.batch_size, self.slot_size]), T.zeros(shape = [len(self.read_heads) + 1, self.batch_size, self.memory_slots]), None])

        return memory_states, read_vector_states, previous_weight_states, output_states

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
    
    def get_weights(self):
        return [self.weights[key] for key in self.weights]

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

        return key, true_shift, T.addbroadcast(sharpen, 1), T.addbroadcast(strengthen, 1), T.addbroadcast(interpolation, 1)

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
    def get_weights(self):
        return [self.weights[key] for key in self.weights]

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

        return key, add, erase, true_shift, T.addbroadcast(sharpen, 1), T.addbroadcast(strengthen, 1), T.addbroadcast(interpolation, 1)

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

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / (gradient_scaling + 1e-10)
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

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
    # </readHead TEST> PASSED 

    # <NTM TEST>
    class Controller:
        def __init__(self):
            self.size = 20

            self.weight_1 = init_weight(15, 20)
        
        def get_weights(self):
            return [self.weight_1]
        
        def forward(self, inp):
            return T.dot(inp, self.weight_1)

    ntm = NTM(controller = Controller(), output_size = 10, memory_slots = 5, slot_size = 2, read_heads = 1, batch_size = 3)

    #external_input, memory, read_vectors, previous_weights
    # five read heads, both reading a vector of size 2, add ten, only leaves room for 5 external input nodes
    e_i = T.matrix() 
    ext_inp = np.random.randn(3, 13) * 0.1

    # batchsize x slot_size x memory_slots (batchsize x N x M)
    m = T.tensor3()
    mem = np.random.randn(3, 2, 5) * 0.1

    # read_heads x batchsize x slot_size
    r_v = T.tensor3()
    read_vec = np.zeros([1, 3, 2])

    # read_heads + write_head x batchsize x memory_slots
    p_w = T.tensor3()
    previous_w = np.zeros([2, 3, 5])

    step = ntm.process_step(e_i, m, r_v, p_w)

    f = theano.function([e_i, m, r_v, p_w], step)
    answer = f(ext_inp, mem, read_vec, previous_w)
    for a in answer:
        print (a.shape)

    # tt x batchsize x data_size
    data = T.tensor3()
    f = theano.function([data], ntm.process(data))

    answer = f(np.zeros(shape = [20, 3, 13]) + 15)
    for a in answer:
        print(a.shape)

    outputs = answer[-1]
    outputs = outputs[:, 0, :]


def main():
    # The big cheese, we're doin it guys!

    # COPY TASK, copy a 20 length tensor of random numbers. Each section will contain 5 bits, one of which (bits 0 - 3) will be on.
    # We'll input the 20 length, then input an end token, which will be the 5th bit (or 4th if using zero-based indexing), then input another 20 length with all zeros. This signals to the NTM that it needs to start outputting the copy. 

    class Controller:
        def __init__(self):
            # Output size is 5, because it needs to output the copied 5 bits 
            self.size = 128

            # We'll have 1 read head, which produces a single read_vector of size 10. We also need to feed in the input, which is of size 5 (for the five bits)
            # so our total input size is 15
            self.fc_1 = init_weight(15, 128)
            self.fc_2 = init_weight(128, 128)

            # This is our controller output 
            self.fc_3 = init_weight(128, 128)

        def get_weights(self):
            return [self.fc_1, self.fc_2, self.fc_3]

        def forward(self, inp):
            fc1 = T.nnet.relu(T.dot(inp, self.fc_1))
            fc2 = T.nnet.relu(T.dot(fc1, self.fc_2))

            # I would ReLU the output, but I already did in the NTM implementation
            fc3 = T.dot(fc2, self.fc_3)
            
            return fc3 

    # output size is 5, for the 5 copy bits
    ntm = NTM(controller = Controller(), output_size = 5, memory_slots = 32, slot_size = 10, read_heads = 1, batch_size = 10)

    data = T.tensor3()
    target = T.tensor3()

    memory_states, _, _, ntm_outputs = ntm.process(data)
    
    # We average the loss across batches, so that we have a singular loss for each timestep. We then average these losses
    # ntm_outputs - target ** 2 -> ts x batchsize x bits
    # 

    loss = T.mean(T.mean(T.sum(.5 * (ntm_outputs - target) ** 2, axis = 2), axis = 1), axis = 0)

    updates = RMSprop(cost = loss, params = ntm.weights, lr = 1e-3)
    
    train = theano.function(inputs = [data, target], outputs = list(ntm.process(data)) + [loss], updates = updates)

    # let's feed a test example
    # ts x batchsize x bits

    end = np.zeros([1, 10, 5])
    for batch in range (10):
        end[0, batch, -1] = 1           # Make the last bit in each batch a 1

    for example in range (100):
        # Produce the first half
        first_half = np.random.randn(20, 10, 5) > 0

        for batch in range (10):
            first_half[:, batch, -1] = 0        # Make sure the last bit (end bit) of each batch is 0

        # Produce second half
        second_half = np.zeros([20, 10, 5])     # Just a bunch of zeros 

        data = np.concatenate([first_half, end, second_half], axis = 0)
        target = np.concatenate([second_half, end, first_half], axis = 0)

        # lamar gotta have that extra timestep for the end bit
        outputs = train(data, target)

        print("LOSS " + str(outputs[-1]))

        outputs = outputs[-2]
        outputs = outputs[:, 0, :]
        
        #plt.imshow(outputs)
        #plt.show()
    

        """
        fig = plt.figure()
        fig.add_subplot(2, 2, 1) 
        plt.imshow(data[:, 0, :], origin = [0, 0])
        fig.add_subplot(2, 2, 2)
        plt.imshow(target[:, 0, :], origin = [10, 0])
        plt.show()
        """

#test()
main()


