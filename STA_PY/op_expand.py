import numpy as np
import random as rd

"""
函数描述：利用伸缩变换公式产生新解，物理上是对 Best 的每个维度都有可能伸缩到（-∞，+∞），然后进行约束到定义域边界，属于全局搜索
参数：
    Best  - 历史最优解
    SE    - 每个算子产生候选解的个数
    gamma - 产生新解时所需参数，叫伸缩因子
返回值：
    y - 产生的SE个新解(SE,N) 
"""
def op_expand(Best,SE,gamma):
	n = Best.size           # 数组中元素的个数(维度),不能用len，因为下一个best为矩阵形式，len[[1,2]] = 1
	Best = Best.reshape(n,1)
	a = np.tile(Best,SE)
	b = np.array([rd.gauss(0,1) for _ in range(n*SE)]).reshape(n,SE)
	y = a + gamma * b * a
	y = y.transpose()
	return y

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	Best = np.array([2,2])
	SE = 1000
	gamma = 1
	state = op_expand(Best,SE,gamma)
	print(state)
	plt.plot(state[:,0],state[:,1],'g*')
	plt.plot(Best[0],Best[1],'or-')
	plt.axis('equal')
	plt.show()
