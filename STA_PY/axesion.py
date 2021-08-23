import numpy as np
from op_axes import op_axes
from translate import translate
from fitness import fitness

"""
函数描述：利用 axes 算子产生新解 并比较这些新解和历史最优解，从而找出最小的解返回
参数：
    funfcn     - 需要优化的函数名
    Best       - 历史最优解
    fBest      - 历史最优值
    SE         - 每个算子产生候选解的个数
    Range      - 各维度的取值范围
    beta       - op_translate 算子产生新解时所需参数，叫平移因子
    delta      - op_axes  算子产生新解时所需参数 ，叫轴向因子
返回值：
    Best    - 迭代优化得到的最优解
    fBest   - 对应的最优值
"""
def axesion(funfcn,Best,fBest,SE,Range,beta,delta):
	Pop_Lb = np.tile(Range[0],(SE,1))     # 定义域下界
	Pop_Ub = np.tile(Range[1],(SE,1))     # 定义域上界
	oldBest = Best                        # 历史最优解

	State = op_axes(Best,SE,delta)        # 利用 axes 算子产生新解

	"""以下四行是对超出定义域的数据进行约束"""
	changeRows = State > Pop_Ub
	State[changeRows] = Pop_Ub[changeRows]
	changeRows = State < Pop_Lb
	State[changeRows] = Pop_Lb[changeRows]

	newBest,fGBest = fitness(funfcn,State)  # 评价这些新解
	if fGBest < fBest:                      # 如果新解中有比历史最优解还好的解，就利用 translate算子看能不能找到更好的解更新，找不到的话 translate 返回是 newBest，具体可以看translate
		Best, fBest =  translate(funfcn,oldBest, newBest, fGBest, SE, Range, beta)
	return Best,fBest


if __name__ == '__main__':
	from Benchmark import Sphere

	funfcn = Sphere
	Best = np.array([2,2])
	fBest = funfcn(Best)
	SE = 4
	n = len(Best)
	Range = np.tile([[-30],[30]],n)
	delta = 1
	beta = 1

	Best,fBest = axesion(funfcn,Best,fBest,SE,Range,beta,delta)
	print("最优解：", Best)
	print("最优值：", fBest)