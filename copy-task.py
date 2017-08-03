import theano
import theano.tensor as T

import numpy as np

from six.moves import cPickle

from matplotlib import pyplot as plt

from heads import writeHead, readHead
from utils import softplus, sigmoid, init_weight, focus_shift, circular_convolution, pass_or_fail, RMSprop 

from memory import mem_read, mem_write, mem_focus

from ntm import NTM

def main():
    train = True
    weight_path = "" #"model-constant-mem_4500.save"

    if train:
        # The big cheese, we're doin it guys!
        
        plt.ion()
        fig = plt.figure()

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
        ntm = NTM(controller = Controller(), output_size = 5, memory_slots = 20, slot_size = 10, read_heads = 1, batch_size = 10)

        data = T.tensor3()
        target = T.tensor3()

        #r = theano.shared(np.random.randn(10, 10, 20))
        r = theano.shared(1.)
        r_ = theano.shared(np.zeros([10, 10, 20])) + r

        if weight_path != '':
            print("loading weights")

            # Load weights, but just the NTM weights, not memory, since we may extend it
            checkpoint = open(weight_path, 'rb')
            
            all_weights = ntm.weights
            for w in all_weights:
                w.set_value(cPickle.load(checkpoint).get_value())
            checkpoint.close()

        memory_states, _, weightings, ntm_outputs = ntm.process(data, r_)
        
        # We average the loss across batches, so that we have a singular loss for each timestep. We then average these losses
        # ntm_outputs - target ** 2 -> ts x batchsize x bits
        # 

        loss = T.sum(T.mean(T.sum(5 * (T.nnet.sigmoid(ntm_outputs) - target) ** 2, axis = 2), axis = 1), axis = 0)

        updates = RMSprop(cost = loss, params = ntm.weights + [r], lr = 1e-3)
        
        train = theano.function(inputs = [data, target], outputs = [memory_states, weightings, weightings, ntm_outputs, loss, updates[2][1]], updates = updates)

        for example in range (5000):
            # Produce the first half
            
            # let's feed a test example
            # ts x batchsize x bits

            end = np.zeros([1, 10, 5])
            for batch in range (10):
                end[0, batch, -1] = 1           # Make the last bit in each batch a 1

            first_half = (np.random.randn(10, 10, 5) > 0).astype(np.float32) * 1

            for batch in range (10):
                first_half[:, batch, -1] = 0        # Make sure the last bit (end bit) of each batch is 0

            # Produce second half
            second_half = np.zeros([10, 10, 5])     # Just a bunch of zeros 

            data = np.concatenate([first_half, end, second_half], axis = 0)
            target = np.concatenate([second_half, end, first_half], axis = 0)

            # lamar gotta have that extra timestep for the end bit
            outputs = train(data, target)

            print("LOSS " + str(outputs[-2]) + ", " + str(example))

            read = outputs[2]
            read = read[:, 0, 0, :]

            write = outputs[2]
            write = write[:, 1, 0, :]

            outputs = outputs[3]
            outputs = outputs[:, 0]

            #.transpose([1, 0])
            
            if (example % 20 == 0 and example != 0):
                cmap = 'jet'

                fig.add_subplot(2, 2, 1)
                plt.imshow(sigmoid(outputs), cmap = cmap)
                fig.add_subplot(2, 2, 2)
                plt.imshow(target[:, 0], cmap = cmap)
                fig.add_subplot(2, 2, 3)
                plt.imshow(read, cmap = cmap)
                fig.add_subplot(2, 2, 4)
                plt.imshow(write, cmap = cmap)
                plt.pause(0.1)

            if (example % 500 == 0):
                print ("SAVING WEIGHTS")
                f = open('model-constant-mem_' + str(example) + '.save', 'wb')
                for w in ntm.weights + [r]:
                    cPickle.dump(w, f, protocol=cPickle.HIGHEST_PROTOCOL)
                
                f.close()
        

            """
            fig = plt.figure()
            fig.add_subplot(2, 2, 1) 
            plt.imshow(data[:, 0, :], origin = [0, 0])
            fig.add_subplot(2, 2, 2)
            plt.imshow(target[:, 0, :], origin = [10, 0])
            plt.show()
            """
    else:
        # Test time!

        plt.ion()
        fig = plt.figure()

        # COPY TASK, we're going to see how well our Neural Turing Machine model extends to longer sequences

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
        ntm = NTM(controller = Controller(), output_size = 5, memory_slots = 80, slot_size = 10, read_heads = 1, batch_size = 10)

        data = T.tensor3()

        # Load weights
        checkpoint = open('pretrained-models-copy/model-constant-mem_4500.save', 'rb')
        
        all_weights = ntm.weights
        for w in all_weights:
            w.set_value(cPickle.load(checkpoint).get_value())

        r = theano.shared(np.zeros(shape = [10, 10, 80]) + cPickle.load(checkpoint).get_value())

        checkpoint.close()

        memory_states, _, weightings, ntm_outputs = ntm.process(data, r)

        test = theano.function(inputs = [data], outputs = [memory_states, weightings, weightings, ntm_outputs])

        for example in range (5000):
            print(r.get_value())

            # Produce the first half
            
            # let's feed a test example
            # ts x batchsize x bits

            end = np.zeros([1, 10, 5])
            for batch in range (10):
                end[0, batch, -1] = 1           # Make the last bit in each batch a 1

            first_half = (np.random.randn(60, 10, 5) > .7).astype(np.float32) * 1

            for batch in range (10):
                first_half[:, batch, -1] = 0        # Make sure the last bit (end bit) of each batch is 0

            # Produce second half
            second_half = np.zeros([60, 10, 5])     # Just a bunch of zeros 

            data = np.concatenate([first_half, end, second_half], axis = 0)

            # lamar gotta have that extra timestep for the end bit
            outputs = test(data)

            read = outputs[2]
            read = read[:, 0, 0, :]

            write = outputs[2]
            write = write[:, 1, 0, :]

            outputs = outputs[3]
            outputs = outputs[:, 0]

            #.transpose([1, 0])
            
            cmap = 'jet'

            fig.add_subplot(2, 2, 1)
            plt.imshow(sigmoid(outputs), cmap = cmap)
            fig.add_subplot(2, 2, 2)
            plt.imshow(data[:, 0], cmap = cmap)
            fig.add_subplot(2, 2, 3)
            plt.imshow(read, cmap = cmap)
            fig.add_subplot(2, 2, 4)
            plt.imshow(write, cmap = cmap)
            plt.pause(0.1)

            input("")

main()
