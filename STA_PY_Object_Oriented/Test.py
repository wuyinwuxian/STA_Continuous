# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:57:13 2021

@author: 中南大学自动化学院  智能控制与优化决策课题组
"""

from STA import STA
import Benchmark
import numpy as np
import matplotlib.pyplot as plt

#参数设置
funfcn = Benchmark.Sphere
Dim = 100
Range = np.repeat([-30,30],Dim).reshape(-1,Dim)
SE = 30
Maxiter = 1000
dict_opt = {'alpha':1,'beta':1,'gamma':1,'delta':1}
# 算法调用
sta = STA(funfcn,Range,SE,Maxiter,dict_opt)
sta.run()
# 画图
#x = np.arange(1,Maxiter+2).reshape(-1,1)
y = sta.history
x = np.arange(y.size).reshape(-1,1)
plt.semilogy(x,y,'b-o')
plt.xlabel('Ierations')
plt.ylabel('fitness')
plt.show()