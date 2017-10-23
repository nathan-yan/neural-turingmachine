import theano
import theano.tensor as T

import numpy as np

def softplus(x):
    """
        softplus(x) -> result (any shape)

        applies softplus to x

        @param x: the value we want to apply the softplus function to
    """

    result = 1 + T.log(1 + T.exp(x))
    return result

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def init_weight(input_size, output_size):
    return theano.shared((np.random.randn(input_size, output_size) * np.sqrt(3 / (input_size + output_size))))

def focus_shift(weight, shift, memory_slots):
    """
        focus_shift(weightings, shift) -> new_weighting

        `circular_convolution` produces the value of the new_weighting for a single memory position. focus_shift applies the circular_convolution to every memory slot using scan.

        @param weight: a batchsize x M matrix that represents the weight vectors
        @param shift: a batchsize x M matrix that represents the shift vectors
    """

    # memory_size -> 1, represents the size of weight -> batchsize x M in its last dimension (M)
    memory_size = weight.shape[-1]

    # new_weighting -> M x batchsize, but will be transposed to have the proper ordering
    new_weighting, updates = theano.scan(fn = circular_convolution, sequences = [T.arange(memory_size)], non_sequences = [shift, weight, memory_slots])

    return new_weighting.dimshuffle([1, 0])

def circular_convolution(weight_idx, shift, weight, memory_slots):
    """
        circular_convolution(weight_idx, shift, weighting) -> column (batchsize)
        
        This function follows the circular convolution rule: column[i] += old_weight[j] * shift[(i - j) % memory_slots], for any particular i. `column` has a batchsize, so `column` will actually be a batchsize column vector. 

        @param weight_idx: a scalar that represents the column index this function is producing, or, in the above example, i
        @param shift: a batchsize x M matrix that represents the shift vectors
        @param weight: a batchsize x M matrix that represents the weight vectors 
    """

    def conv(shift_idx, current_value, shift, weight_idx, weight, memory_slots):
        current_value += weight[:, shift_idx] * shift[:, (weight_idx - shift_idx)%memory_slots]

        return current_value

    # memory_size -> 1, represents the size of weight -> batchsize x M in its last dimension (M)
    memory_size = weight.shape[-1]

    # batch_size -> 1, represents the size of weight -> batchsize x M in its first dimension (batchsize)
    batch_size = weight.shape[0]

    # columns -> timesteps x batchsize
    columns, updates = theano.scan(fn = conv, sequences = [T.arange(memory_size)], outputs_info = [T.zeros(shape = [batch_size])], non_sequences = [shift, weight_idx, weight, memory_slots])

    # column -> batchsize
    column = columns[-1]
    return column

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