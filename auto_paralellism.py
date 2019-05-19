#!/usr/bin/env python
# coding: utf-8

# In[1]:


import d2lzh as d2l
import mxnet as mx
from mxnet import nd


# In[2]:


def run(x):
    return [nd.dot(x, x) for _ in range(10)]


# In[ ]:


x_cpu = nd.random.uniform(shape=(2000, 2000))
x_gpu = nd.random.uniform(shape=(6000, 6000), ctx = mx.gpu(0))


# In[ ]:


run(x_cpu)
run(x_gpu)
nd.waitall()

with d2l.Benchmark('Run on CPU.'):
    run(x_cpu)
    nd.waitall()
with d2l.Benchmark('Then run on GPU.'):
    run(x_gpu)
    nd.waitall()
    


# In[ ]:


#自动并行计算
with d2l.Benchmark('Run on both CPU and GPU in parallel.'):
    run(x_cpu)
    run(x_gpu)
    nd.waitall()
    


# In[ ]:


def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]
with d2l.Benchmark('Run on GPU.'):
    y = run(x_gpu)
    nd.waitall()
with d2l.Benchmark('Then copy to CPU.'):
    copy_to_cpu(y)
    nd.waitall()
    


# In[ ]:


with d2l.Benchmark('Run and copy in parallel.'):
    y = run(x_gpu)
    copy_to_cpu(y)
    nd.waitall()

