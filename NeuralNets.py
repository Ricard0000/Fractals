#
#This code is a modification of Maziar Raissi's code, available on Github.
#It is from his paper on "Physics Informed Deep Learning"
#

#Here I use use bits of Maziar's code to do function approximations with Neural Networks.

#It is not a code for solving partial differential equations. I also modified it so that

#the weights and biases are saved as numpy array variables. These are accessible by

#typing "weights" or "biases"into the console.

#
#This is a python

#The Neural Network is different in structure from the above MATLAB code.

#The nonlinear activation function is also different.

#


import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

class Fit_to_function:
    # Initialize the class
    def __init__(self, x0, u0, layers):

        self.x0 = x0
        self.u0 = u0
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

        # tf Graphs
        self.u0_pred, self.weights_tf, self.b_tf = self.net_uv(self.x0_tf)
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred))
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - -1.0)/(1.0 - -1.0) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_uv(self, x):        
        u = self.neural_net(x, self.weights, self.biases)
        
        return u, self.weights, self.biases
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0,
                   self.u0_tf: self.u0}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)    
            # Print

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,         
                                fetches = [self.loss],
                                loss_callback = self.callback)
    def predict(self, x0):        
        tf_dict = {self.x0_tf: x0}
        u_star = self.sess.run(self.u0_pred, tf_dict)       
        biases=self.sess.run(self.b_tf)
        weights=self.sess.run(self.weights_tf)
        return u_star, weights, biases
    
if __name__ == "__main__":

    Nx = 50
    layers = [1, 5, 5, 5, 5, 1]          
    a=-1.0
    b=1.0
    
    x0=np.zeros([Nx,1], dtype=float)   
    u0=np.zeros([Nx,1], dtype=float)
    for I in range(0,Nx):
        x0[I,0]=a+(b-a)/(Nx-1)*(I)
        u0[I,0]=x0[I,0]**2.0
    
    model = Fit_to_function(x0, u0, layers)
             
    start_time = time.time()
    model.train(1000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
        
    u_pred, weights, biases= model.predict(x0)
    h_pred = np.sqrt(u_pred**2)        
    
    plt.scatter(x0,u_pred, color='black', linewidth=3, label='Prediction')
    plt.plot(x0,u0, color='red', linewidth=3, label='Exact')
    plt.show()

    plt.plot(x0,u0-u_pred,color='black', linewidth=3, label='Error')
    plt.show()