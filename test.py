import theano
import theano.tensor as T

import numpy as np

from utils import softplus, sigmoid, init_weight, focus_shift, circular_convolution, pass_or_fail, RMSprop 

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

test()