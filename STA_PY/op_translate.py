import numpy as np


"""
函数描述：利用平移公式产生新解，物理上是在 oldBest —— newBest 这条直线上进行搜索产生新解，我们认为新旧两个最优解的连线上大概率会出现好的解，比如两个解在谷底两侧时
参数：
    oldBest  - 历史最优解
    newBest  - 新的最优解
    SE       - 每个算子产生候选解的个数
    beta     - 产生新解时所需参数，叫平移因子
返回值：
    y - 产生的SE个新解(SE,N) 
"""
def op_translate(oldBest,newBest,SE,beta):
	n = oldBest.size
	oldBest = oldBest.reshape(n,1)
	newBest = newBest.reshape(n,1)    # 定义局部变量
	diff = (newBest - oldBest)
	a = np.tile(newBest,SE)
	b = beta/(np.linalg.norm(diff) + 2e-16) # 需要加上一个极小值
	c = np.tile(np.random.uniform(0,1,(1,SE)),n).reshape(n,SE)* np.tile(diff,SE)
	y = a + b * c
	y = y.transpose()
	return y

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	oldBest = np.array([1,1])
	newBest = np.array([2,2])
	SE = 200
	beta = 1
	state = op_translate(oldBest,newBest,SE,beta)
	print(state)
	plt.plot(state[:,0],state[:,1],'g*-') # 画出各点
	Best = np.vstack((oldBest,newBest))  # 组合   newBest 和Best 仍然没有变，为行向量
	plt.plot(Best[:,0],Best[:,1],'ro-')  # 画出两点连线
	plt.plot(oldBest[0],oldBest[1],'ro-')# 画出旧点
	plt.plot(newBest[0],newBest[1],'ro-')# 画出新点
	plt.show()
