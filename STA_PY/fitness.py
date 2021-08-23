import numpy as np


"""
函数描述：利用 expand 算子产生新解 并比较这些新解和历史最优解，从而找出最小的解返回
参数：
    funfcn     - 需要优化的函数名
    state      - 解构成的矩阵，在STA算法中解叫做状态，所以也叫状态矩阵
返回值：
    Best     - state 所有解中最好的一个解
    fGBest   - 其对应的最优值
"""
def fitness(funfcn,State):
	fState = list(map(funfcn,State))     # 调用
	fGBest = np.min(fState)
	Best = State[fState.index(fGBest)]   # 这个列表中第一次此值的索引
	return Best,fGBest


if __name__ == '__main__':
	from op_rotate import op_rotate
	from Benchmark import Sphere

	funfcn = Sphere
	Best = np.array([2,2])
	SE = 4
	alpha = 1
	State = op_rotate(Best,SE,alpha)
	print(State)
	newBest,fGBest = fitness(funfcn,State)
	print(newBest,fGBest)