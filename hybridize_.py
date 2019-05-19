#!/usr/bin/env python
# coding: utf-8

# In[5]:


#符号式编程,先返回计算流程，最后给定输入后，调用编译好的程序执行
def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''
def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)


# In[6]:


#在HybridBlock或HybridSequential下调用hybridize函数，可以转换为符号式编程
from mxnet import nd, sym
from mxnet.gluon import nn
import time

def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256, activation='relu'),
           nn.Dense(128, activation='relu'),
           nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)


# In[7]:


net.hybridize()
net(x)           #计算速度加快


# In[15]:


#比较计算速度
def benchmark(net, x):
    start = time.time()
    for i in range(100):
        _ = net(x)
    nd.waitall()
    return time.time() - start

net = get_net()
print('before hybridzing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('after hybridng: %.4f sec' % (benchmark(net, x)))


# In[27]:


#用HybridBlock类构造模型
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)
        
    def hybrid_forward(self, F, x):
            print('F: ', F)
            print('x: ', x)
            x = F.relu(self.hidden(x))
            print('hidden: ',x)
            return self.output(x)


# In[28]:


net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
net(x)


# In[29]:


net.hybridize()
net(x)


# In[12]:


net(x)#再一次运行不会print，因为上一次运行之后符号式程序已经得到，再次访问不会读取python代码
#直接在C++后端执行符号式程序


# In[ ]:




