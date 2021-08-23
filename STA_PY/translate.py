import numpy as np
from op_translate import op_translate
from fitness import fitness

"""
函数描述：利用 translate 算子产生新解 并比较这些新解和历史最优解，从而找出最小的解返回
参数：
    funfcn     - 需要优化的函数名
    oldBest    - 历史最优解
    Best       - expand,axesion,rotate算子得到的最优解
    fBest      - 历史最优值
    SE         - 每个算子产生候选解的个数
    Range      - 各维度的取值范围
    beta       - op_translate 算子产生新解时所需参数，叫平移因子
返回值：
    Best    - 迭代优化得到的最优解
    fBest   - 对应的最优值
"""
def translate(funfcn,oldBest,Best,fBest,SE,Range,beta):
    Pop_Lb = np.tile(Range[0], (SE, 1))           # 定义域下界
    Pop_Ub = np.tile(Range[1], (SE, 1))           # 定义域下界

    State = op_translate(oldBest,Best,SE,beta)    # 利用 translate 算子产生新解

    """以下四行是对超出定义域的数据进行约束"""
    changeRows = State > Pop_Ub
    State[changeRows] = Pop_Ub[changeRows]
    changeRows = State < Pop_Lb
    State[changeRows] = Pop_Lb[changeRows]

    newBest,fGBest = fitness(funfcn,State)       # 评价这些新解
    if fGBest < fBest:
        fBest,Best = fGBest,newBest
    return Best, fBest