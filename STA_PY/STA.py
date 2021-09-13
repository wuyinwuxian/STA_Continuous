import numpy as np
from rotate import rotate
from expand import expand
from axesion import axesion


"""
函数描述：利用STA算法对函数进行优化，benchmark 都是无约束的
参数：
    funfcn     - 需要优化的函数名
    Best       - 历史最优解
    SE         - 每个算子产生候选解的个数
    Range      - 各维度的取值范围
    Iterations - 迭代次数
返回值：
    Best    - 迭代优化得到的最优解
    fBest   - 对应的最优值
    history - 历史最优值的的变化过程矩阵
"""
def STA(funfcn,Best,SE,Range,Iterations):
	""" 算法初始化 """
	alpha_max = 1
	alpha_min = 1e-4
	alpha = alpha_max
	beta = 1
	gamma = 1
	delta = 1
	fc = 2
	history = np.empty((Iterations,1))
	fBest = funfcn(Best[0])

	for iter in range(Iterations):   	# 迭代优化
		Best,fBest = expand(funfcn,Best,fBest,SE,Range,beta,gamma)    # 利用 expand 算子进行迭代优化
		Best,fBest = rotate(funfcn,Best,fBest,SE,Range,alpha,beta)    # 利用 rotate 算子进行迭代优化
		Best,fBest = axesion(funfcn,Best,fBest,SE,Range,beta,delta)   # 利用 axesion 算子进行迭代优化
		history[iter] = fBest
		alpha = alpha/fc if alpha > alpha_min else alpha_max

	return Best,fBest,history

if __name__ == '__main__':
	from Benchmark import Sphere

	funfcn = Sphere
	SE = 10
	Dim = 10
	Range = np.repeat([-30,30],Dim).reshape(2,Dim)
	# print(Range)
	Best0 = Range[0,:] + (Range[1,:]-Range[0,:]*np.random.uniform(0,1,(1,Dim)))
	print("初始解：", Best0)

	Iterations = 1000
	Best,fBest,history = STA(funfcn,Best0,SE,Range,Iterations)
	print("最优解：",Best,"\n最优值：",fBest)