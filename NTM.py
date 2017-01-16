import theano
import theano.tensor as T 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.tensor.signal import conv

import numpy as np

from weights import init_weights, init_weights_conv

import time
import sys

class head:
    def __init__(self, input_size, memory_locations, memory_size, shifts_allowed = 1):
        self.w_key = init_weights(input_size, memory_size) 
        self.w_strength = init_weights(input_size, 1)
        self.w_interpolation = init_weights(input_size, 1)
        self.w_shift = init_weights(input_size, memory_locations)
        self.w_sharpen = init_weights(input_size, 1)

        self.params = [self.w_key, self.w_strength, self.w_interpolation, self.w_shift, self.w_sharpen]

        self.shifts_allowed = shifts_allowed
        self.input_size = input_size 

        self.memory_locations = memory_locations
        self.memory_size = memory_size 
    
    def forward(self, controller_out):
        # key - batchsize x memory_size
        # strength - batchsize x 1 
        # interpolation - batchsize x 1 
        # shift - batchsize x (self.shifts_allowed * 2 + 1)

        key = T.dot(controller_out, self.w_key)
        strength = T.dot(controller_out, self.w_strength)
        interpolation = T.nnet.sigmoid(T.dot(controller_out, self.w_interpolation))
        shift = T.nnet.softmax(T.dot(controller_out, self.w_shift))
        sharpen = T.dot(controller_out, self.w_sharpen)

        return key, strength, interpolation, shift, sharpen 

class NTM:
    def __init__(self, batchsize, controller_input, controller_hidden, controller_output, memory_locations, memory_size, read_heads, write_heads = 1, shifts_allowed = 1):
        sys.stdout.write("Initializing weights...")
        start = time.time()

        self.cn_w1 = init_weights(controller_input, controller_hidden) 
        self.cn_w2 = init_weights(controller_hidden, controller_output) 
        self.cn_w_ = init_weights(memory_size * read_heads, controller_hidden)

        sys.stdout.write('.' * (50 - len("Initializing weights...")) + str(time.time() - start) + '\n')

        self.shifts_allowed = shifts_allowed

        sys.stdout.write("Initializing heads...")
        start = time.time()

        self.read_heads = read_heads 
        self.read_h = [head(input_size = controller_hidden, memory_locations = memory_locations, memory_size = memory_size, shifts_allowed = 1) for h in range (self.read_heads)]

        self.write_heads = write_heads
        self.write_h = head(input_size = controller_hidden, memory_locations = memory_locations, memory_size = memory_size, shifts_allowed = 1)

        sys.stdout.write('.' * (50 - len("Initializing heads...")) + str(time.time() - start) + '\n')

        self.controller_input = controller_input
        self.controller_hidden = controller_hidden
        self.controller_output = controller_output

        sys.stdout.write("Initializing memory...")
        start = time.time()

        self.memory_locations = memory_locations
        self.memory_size = memory_size

        self.batchsize = batchsize 

        self.E = np.ones(shape = (self.batchsize, self.memory_locations, self.memory_size))
        self.shift_template = np.zeros(shape = (self.memory_locations, self.batchsize))

        self.erase_params = init_weights(controller_hidden, self.memory_locations)
        self.add_params = init_weights(controller_hidden, self.memory_locations)

        self.params = []

        self.params += self.write_h.params 

        for h in self.read_h: 
            self.params += h.params 
        
        self.params.append(self.erase_params)
        self.params.append(self.add_params)
        self.params += [self.cn_w1, self.cn_w2, self.cn_w_]

        sys.stdout.write('.' * (50 - len("Initializing memory...")) + str(time.time() - start) + '\n')
        sys.stdout.write("Initializing feedforward...")
        start = time.time()

        def read(memory, w_r):
            # w_r will have shape batchsize x memory_loc 
            # dot with w_r and memory will yield a batchsize x memory_size matrix, giving the averaged sum of the memory based on the weight 

            r_t = T.batched_dot(w_r, memory) 
            return r_t 
            
        def write(memory, w_w, e, v):
            M_ = T.mul(memory, (self.E - T.batched_dot(w_w.dimshuffle([0, 1, 'x']), e.dimshuffle([0, 'x', 1]))))
            M = M_ + T.batched_dot(w_w.dimshuffle([0, 1, 'x']), v.dimshuffle([0, 'x', 1]))

            return M
        
        def cosine(u, v):
            #T.batched will have shape batchsize x mem_loc
            # T.sum(v) will have shape batchsize x 1* 
            # T.sum(u) will have shape batchsize x mem_loc

            dot = T.batched_dot(u, v)/T.sqrt(T.sum(u ** 2, axis = 2) * T.sum(v ** 2, axis = 1).dimshuffle([0, 'x']))

            return dot # Will return an array of size batchsize x mem_loc 

        def C(memory, key, strength):
            key_fit = cosine(memory, key)   # Will have size batchsize x mem_loc
            key_fit *= T.addbroadcast(strength, 1)  # Will have size batchsize x 1 

            return T.nnet.softmax(key_fit)  # Will return a matrix of size batchsize x mem_loc

        def circular_convolution(weight, shift):
            concat = np.asarray([weight.reshape((1, -1)), shift.reshape((1, -1))])
            bs,w=concat[0].shape # [batch_size,width]
            corr_expr = T.signal.conv.conv2d(
                concat[0], 
                concat[1][::-1].reshape((1, -1)), # reverse the second vector
                image_shape=(1, w), 
                border_mode='full')
            corr_len = corr_expr.shape[1]
            pad = w - corr_len%w    
            v_padded = T.concatenate([corr_expr, T.zeros((bs, pad))], axis=1)
            circ_corr_exp = T.sum(v_padded.reshape((bs, v_padded.shape[1] // w, w)), axis=1)

            return circ_corr_exp[:, ::][0]

        def shift(weights, shift_vectors):
            [(weights), updates] = theano.scan(fn = circular_convolution, sequences = [weights, shift_vectors])  

            return weights.reshape((50, 30))

        def forward(ext_inp, target, read_vectors, memory, w_rt_1, w_wt_1, loss):
            # CONTROLLER 
            r_fc = T.dot(read_vectors, self.cn_w_) # batchsize x controller_hidden
            i_fc = T.dot(ext_inp, self.cn_w1)   # batchsize x controller_hidden

            controller_hidden = r_fc + i_fc    # batchsize x controller_hidden
            controller_hidden = T.nnet.relu(controller_hidden)  # batchsize x controller_hidden

            output = T.dot(controller_hidden, self.cn_w2)   # batchsize x controller_output
            loss += T.mean(.5 * T.sum((output - target) ** 2, axis = 1))  # scalar 

            erase_vector = T.nnet.sigmoid(T.dot(controller_hidden, self.erase_params))  # batchsize x mem_loc
            add_vector = T.dot(controller_hidden, self.add_params)  # batchsize x mem_loc

            # Loop through every head, get their parameters and update memory or reads as needed 
            # We start with write first

            w_key, w_strength, w_interpolation, w_shift, w_sharpen = self.write_h.forward(controller_hidden)
            prelim_weight = C(memory, w_key, w_strength)    # batchsize x mem_loc
            interpolated_weight = T.addbroadcast(w_interpolation, 1) * prelim_weight + \
                                  (1 - T.addbroadcast(w_interpolation, 1)) * w_wt_1  # prelim_weight and                                                 w_wt_1 have size                                                  batchsize x mem_loc

            #shift_vector = np.zeros(shape = (self.memory_locations, self.batchsize))
            #w_shift = T.transpose(w_shift, [1, 0])

            #for i in range (-self.shifts_allowed, self.shifts_allowed + 1, 1):
            #    shift_vector[i] = shift_vector[i] + w_shift[i + self.shifts_allowed] 

                # shift_vector[i] will be of size 1 x batchsize, w_shift[i + self.shifts_allowed] will be of size 1 x batchsize  
            
            #shift_vector will now be of size mem_loc x batchsize 
            shift_vector = w_shift    # now shift_vector is of size batchsize x mem_loc
            
            shifted_weight = shift(interpolated_weight, shift_vector)   # batchsize x mem_loc

            w_sharpen = T.exp(w_sharpen) + 1 
            # interpolated_weight - batchsize x mem_loc, w_sharpen - batchsize x 1
            sharpened_weight = shifted_weight ** T.addbroadcast(w_sharpen, 1) 
            w_sharpened_weight = sharpened_weight/T.sum(sharpened_weight, axis = 1).dimshuffle([0, 'x']) # sum(sharpened_weight) - batchsize

            memory = write(memory, w_sharpened_weight, erase_vector, add_vector)

            reads = []
            weightings = []

            for h in range (self.read_heads):
                w_key, w_strength, w_interpolation, w_shift, w_sharpen = self.read_h[h].forward(controller_hidden)
                prelim_weight = C(memory, w_key, w_strength)
                interpolated_weight = T.addbroadcast(w_interpolation, 1) * prelim_weight + \
                                    T.addbroadcast(w_interpolation, 1) * w_rt_1[h]  # prelim_weight and                                                 w_wt_1 have size                                                  batchsize x mem_loc

                #shift_vector = np.zeros(shape = (self.memory_locations, self.batchsize))
                #w_shift = T.transpose(w_shift, [1, 0])

                #for i in range (-self.shifts_allowed, self.shifts_allowed + 1, 1):
                #    shift_vector[i] += w_shift[i + self.shifts_allowed] 
                    # shift_vector[i] will be of size 1 x batchsize, w_shift[i + self.shifts_allowed] will be of size 1 x batchsize  
                
                #shift_vector will now be of size mem_loc x batchsize 
                shift_vector = w_shift    # now shift_vector is of size batchsize x mem_loc
                
                shifted_weight = shift(interpolated_weight, shift_vector)

                w_sharpen = T.exp(w_sharpen) + 1 
                # interpolated_weight - batchsize x mem_loc, w_sharpen - batchsize x 1
                sharpened_weight = shifted_weight ** T.addbroadcast(w_sharpen, 1) 
                sharpened_weight /= T.sum(sharpened_weight, axis = 1).dimshuffle([0, 'x']) # sum(sharpened_weight) - batchsize
                # end sharpened_weight - batchsize x mem_loc

                weightings.append(sharpened_weight)
                
                reads.append(read(memory, sharpened_weight))
            
            reads = T.concatenate(reads, axis = 1)
            weightings = T.stack(weightings)
            
            return reads, memory, w_sharpened_weight, weightings, loss

        self.inp = T.tensor3() # Sequence x batchsize x seq_len 
        self.targets = T.tensor3()  # Sequence x batchsize x seq_len 

        sys.stdout.write('.' * (50 - len("Initializing feedforward...")) + str(time.time() - start) + '\n')

        sys.stdout.write("Initializing main function...")
        start = time.time() 

        [(read_vectors, memory_states, _, _, loss), updates] = theano.scan(fn = forward, sequences = [self.inp, self.targets], outputs_info = [np.zeros(shape = (self.batchsize, self.read_heads * self.memory_size), dtype=np.float64), np.zeros(shape = (self.batchsize, self.memory_locations, self.memory_size), dtype=np.float64), np.zeros(shape = (self.batchsize, self.memory_locations), dtype=np.float64), np.zeros(shape = (self.read_heads, self.batchsize, self.memory_locations), dtype=np.float64), np.float64(0.)])

        final_loss = loss[-1]
        sys.stdout.write('.' * (50 - len("Initializing main function...")) + str(time.time() - start) + '\n')

        sys.stdout.write("Compiling gradients...")
        start = time.time() 
        updates = RMSprop(cost = final_loss, params = self.params , lr = 0.001)
        sys.stdout.write('.' * (50 - len("Compiling gradients...")) + str(time.time() - start) + '\n')

        start = time.time()
        sys.stdout.write("Compiling train...")
        self.train = theano.function(inputs = [self.inp, self.targets], outputs = [memory_states, final_loss])
        sys.stdout.write('.' * (50 - len("Compiling train...")) + str(time.time() - start) + '\n')

        print "Neural Turing Machine has booted..."

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def main():
    test = NTM(batchsize = 50, controller_input = 8, controller_hidden = 300, controller_output = 8, memory_locations = 30, memory_size = 10, read_heads = 1, write_heads = 1)

    for training_sample in range (100000):
        batch = np.random.uniform(low = -1, high = 1, size = (20, 50, 8)) > 0
        for i in range (20):
            for j in range (50):
                # Ensure that you don't accidentally get a delimiter
                batch[i][j][np.random.randint(0, 6)] = 1    
            
                if i == 19:
                    batch[i][j] = np.zeros(8) 
                    batch[i][j][-1] = 1
        

        blank_space = np.zeros(shape = (20, 50, 8))
        inputs = np.concatenate([batch, blank_space], axis = 0) 
        outputs = np.concatenate([blank_space, batch], axis = 0)
        print inputs.shape

        memory_states, final_loss = test.train(inputs, outputs)
        if training_sample % 10:
            print final_loss

main()