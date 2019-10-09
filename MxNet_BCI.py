# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:32:57 2018

@author: Melvin
"""

import mxnet as mx

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

batch_size = 3
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=False)
val_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)

data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)

# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=1000)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 500)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=2)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')


import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on compute context
mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=50)  # train for at most 10 dataset passes

test_iter = mx.io.NDArrayIter(test_data, None, batch_size)
prob = mlp_model.predict(test_iter)
#assert prob.shape == (10000, 10)

test_iter = mx.io.NDArrayIter(test_data, test_label, batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
#assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]
