#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss 
import time


# In[ ]:


#初始化参数
scale = 0.01
w1 = nd.random.normal(scale=scale, shape=(20, 1, 3, 3))
b1 = nd.zeros(shape=20)
w2 = nd.random.normal(scale=scale, shape=(50, 20, 5, 5))
b2 = nd.zeros(shape=50)
w3 = nd.random.normal(scale=scale, shape=(800, 128))
b3 = nd.zeros(shape=128)
w4 = nd.random.normal(scale=scale, shape=(128, 10))
b4 = nd.zeros(shape=10)
params = [w1, b1, w2, b2, w3, b3, w4, b4]

#定义模型
def lenet(X, params):
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1],
                            kernel=(3,3), num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                   stride=(2, 2))
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3],
                            kernel=(5, 5),num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='avg', kernel=(2,2),
                   stride=(2, 2))
    h2 = nd.flatten(h2)
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = nd.relu(h3_linear)
    y_hat = nd.dot(h3, params[6]) + params[7]
    return y_hat

#定义交叉熵损失函数
loss = gloss.SoftmaxCrossEntropyLoss()


# In[ ]:


#get_params函数将所有模型参数复制到显卡
def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params

new_params = get_params(params, mx.gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)


# In[ ]:


#allreduce函数将各显卡的显存加起来再传给个显卡
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)  #先把所有的结果都加到第一个数据
    for i in range(1, len(data)):
        data[0].copyto(data[i])  #再把第一个数据的结果copy给其他数据列
#测试allreduce函数
data = [nd.ones((1, 2), ctx=ms.gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:', data)
allreduce(data)
print('after allreduce:', data)


# In[ ]:


#把数据样本分给各个gpu
def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k  #假设可以整除
    assert m * k == n, '#example is not divided by # device.'
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]


# In[ ]:


#例，将六个数据平均分给两块显卡
batch = na.arange(24).reshape((6, 4))
ctx = [mx.gpu(0), mx.gpu(1)]
splitted = split_and_load(batch, ctx)
print('input: ', batch)
print('load into', ctx)
print('output:'. splitted)


# In[ ]:


def train_batch(X, y, gpu_params, ctx, lr):
    gpu_Xs, gpu_ys = split_and_load(X, ctx), split_and_load(y, ctx)
    with autograd.record():
        ls = [loss(lenet(gpu_X, gpu_y) 
                for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params))]
    for l in ls:
        l.backward()
    for i in range(len(gpu_params[0])):
        allreduce([gpu_params[c][i].grad for c in range(len(ctx))])
    for param in gpu_params:
        d2l.sgd(param, lr, X.shape[0])


# In[ ]:


def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    gpu_params = [get_params(params, c) for c in ctx]
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            train_batch(X, y, gpu_params, ctx, lr)
            nd.waitall()
        train_time = time.time() - start
        
        def net(x):
            return lenet(x, gpu_params[0])
        
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time %.1f sec, test acc %.2f'
             % (epoch + 1, train_time, test_acc))


# In[ ]:


#单gpu
train(num_gpus=1, batch_size=256, lr=0.2)
#双gpu
train(num_gpus=2, batch_size=256, lr=0.2)

