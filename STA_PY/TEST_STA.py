import numpy as np
import random as rd
import matplotlib.pyplot as plt
from Benchmark import Sphere,Rastrigin,Rosenbrock,Griewank,Michalewicz
from STA import STA

SE = 30
Dim = 5
Range = np.tile([[-30],[30]],Dim)
Iterations = 500
Best0 = np.array(Range[0,:] + (Range[1,:]-Range[0,:]*np.random.uniform(0,1,(1,Dim))))

xmin,fxmin,history = STA(Griewank,Best0,SE,Range,Iterations)  # 利用STA进行优化

print("此函数最小值点:",xmin,'\n',"此函数最小值:",fxmin)
plt.plot(history,'b.-')
# plt.semilogy(history,'b.-') # 对数曲线
plt.xlabel('Iterations')
plt.ylabel('fitness')
plt.show()
