import numpy as np
import random as rd

"""
函数描述：利用轴向变换公式产生新解，物理上是对 Best 的单个维度都有可能伸缩到（-∞，+∞），然后进行约束到定义域边界，属于局部搜索，增强的是单维搜索能力
参数：
    Best  - 历史最优解
    SE    - 每个算子产生候选解的个数
    delta - 产生新解时所需参数，叫轴向因子
返回值：
    y - 产生的SE个新解(SE,N) 
"""
def op_axes(Best,SE,delta):
	n = Best.size
	A = np.zeros((n,SE))
	index = np.random.randint(0,n,(1,SE))
	A[index,list(range(SE))] = 1
	Best = Best.reshape(n,1)
	a = np.tile(Best,SE)
	b = np.array([rd.gauss(0,1) for _ in range(n*SE)]).reshape(n,SE)
	c = delta*b*A*a
	y = a + c
	y = y.transpose()
	return y

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	Best = np.array([2,2])
	SE = 200
	delta = 1
	state = op_axes(Best,SE,delta)
	print(state)
	plt.plot(state[:,0],state[:,1],'g*')
	plt.plot(Best[0],Best[1],'or')
	plt.axis('equal')
	plt.show()
