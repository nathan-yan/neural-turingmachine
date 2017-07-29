shift = [0, 0, 0, 0, 0, 1]
vector = [1, 2, 3, 4, 5, 6]

def circular_convolution(vector, shift):
    new = [0, 0, 0, 0, 0 ,0]
    for i in range (6):
        for j in range (6):
            new[i] += vector[j] * shift[(i - j)%6]
    
    return new 

print(circular_convolution(vector, shift))

import theano
import theano.tensor as T

import numpy as np 

weightings = T.matrix()
shift = T.matrix() 

# 32 is our example memory size
# batchsize is 2 

def circular_convolution_column(weight_idx, shift, weighting):
    def conv(shift_idx, current_value, shift, weight_idx, weighting):
        current_value += weighting[:, shift_idx] * shift[:, (weight_idx - shift_idx)%5]

        return current_value

    columns, updates = theano.scan(fn = conv, sequences = [theano.tensor.arange(5)], outputs_info = [np.zeros(shape = [2])], non_sequences = [shift, weight_idx, weighting])

    return columns[-1]

def circular_convolution(weightings, shift):
    conv, updates = theano.scan(fn = circular_convolution_column, sequences = [theano.tensor.arange(5)], non_sequences = [shift, weightings])

    return conv

f = theano.function([weightings, shift], [circular_convolution(weightings, shift)])

weight = np.array([[0, 0, 0, 0, 1], [1, 2, 3, 4, 5]])
shifts = np.array([[.9, .1, 0, 0, 0], [0, 0, 0, 0, 1]])

print(weight)
print(f(weight, shifts)[0].T) 
# returns shape 5 x 2