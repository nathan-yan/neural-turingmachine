import theano
import theano.tensor as T

import numpy as np

from utils import softplus, sigmoid, init_weight, focus_shift, circular_convolution, pass_or_fail, RMSprop 

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
        return [self.weights["controller->key"], self.weights["controller->shift"], self.weights["controller->sharpen"], self.weights["controller->strengthen"], self.weights["controller->interpolation"]]

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
        return [self.weights["controller->key"], self.weights["controller->add"], self.weights["controller->erase"], self.weights["controller->shift"], self.weights["controller->sharpen"], self.weights["controller->strengthen"], self.weights["controller->interpolation"]]

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
        add = T.tanh(T.dot(controller, self.weights["controller->add"]))
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
