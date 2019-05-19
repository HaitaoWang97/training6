#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
import time


# In[ ]:


#使用残差网络作为训练模型
def resnet18(num-classes):
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i ==1=0 and not first_block:
                blk.add(d2l.Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk
    
    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
           nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
           resnet_block(128, 2),
           resnet_block(256, 2),
           resnet_block(512, 2))
    net.add(nn.GlobalAvPool2D(), nn.Dense(num_classes))
    return net

net = resnet18(10)


# In[ ]:


ctx = [mx.gpu(0), mx.gpu(1)]
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)


# In[ ]:


x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctx)
net(gpu_x[0]), net(gpu_x[1])


# In[ ]:


#训练模型
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('runnig on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and-load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_x), gpu_y)
                     for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        train_time = time.time() - start
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time %.1f sec, test acc %.2f' % (epoch + 1, train_time, test_acc))


# In[ ]:


#单gpu
train(num_gpus=1, batch_size=256, lr=0.1)
#双gpu
train(num_gpus=2, batch_size=512, lr=0.2)

