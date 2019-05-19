#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, nn
import os
import subprocess
import time


# In[5]:


#异步计算，前端命令不用等到后端计算完成就会返回
class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''
        
    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self, *args):
        print("%stime: %.4f sec" % (self.prefix, time.time() - self.start))


# In[6]:


with Benchmark('Workloads are queued.'):
    x = nd.random.uniform(shape=(2000, 2000))
    y = nd.dot(x, x).sum()
    
with Benchmark('Workloads are finished.'):
    print('sum = ', y)


# In[7]:


#使用同步函数等待后端计算结果如wait_to_read,waitall，asnumpy,asscalar
with Benchmark():
    y = nd.dot(x, x)
    y.wait_to_read()
    


# In[9]:


with Benchmark():
    y = nd.dot(x, x)
    z = nd.dot(x, x)
    nd.waitall()


# In[10]:


with Benchmark():
    y = nd.dot(x, x)
    y.asnumpy()
    


# In[11]:


with Benchmark():
    y = nd.dot(x, x)
    y.norm().asscalar()


# In[16]:


#在for循环内使用wait_to_read,每次前端都等待计算完成后返回
with Benchmark('synchronous.'):
    for _ in range(100):
        y = x + 1
        y.wait_to_read()
#在for循环外使用同步函数，前端不等待后端计算完成
with Benchmark('asynchronous.'):
    for _ in range(100):
        y = x + 1
    nd.waitall()


# In[37]:


def data_iter():
    start = time.time()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        X = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 ==0:
            print('batch %d, time %f sec' % (i + 1, time.time() - start))


# In[38]:


net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'),
       nn.Dense(512, activation='relu'),
       nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.005})
loss = gloss.L2Loss()


# In[ ]:





# In[39]:


for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()


# In[40]:


l_sum = 0
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    l_sum += l.mean().asscalar()
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()


# In[ ]:




