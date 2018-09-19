"""
author:louis
date:19/09/2018
function:nothing
version:1.0
"""
import numpy as np
from scipy import sparse
import mglearn
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
def main():

    #creat a 2d numpy array
    eye = np.eye(4)
    print('numpy array:\n{}'.format(eye))

    #将numpy数组转换为csr格式的scipy稀疏矩阵
    #只保存非零元素
    sparse_matrix = sparse.csr_matrix(eye)
    print('\nscipy sparse CSR matrix : \n{}'.format(sparse_matrix))
    #########

    #在-10到10生成一共数列.共一百个
    x = np.linspace(-10, 10, 100)
    #用since创建第二个数组
    y = np.sin(x)
    # plot 函数绘制一个数组关于另外一个数组的折线图
    plt.plot(x, y, marker = 'x')
    plt.show()
    #########

    #创建关于个人的简单数据集
    data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
            'Location': ['NewYork', 'Paris', 'Berlin', 'London'],
            'Age': [24, 13, 53, 33]
            }
    data_pandas = pd.DataFrame(data)
    # Ipython.display 可以 在jupyter notebook 中打印出 '美观的' dataframe
    display(data_pandas)

    #选择年龄大于30的所有行
    display(data_pandas[data_pandas.Age > 30])
if __name__ == '__main__':
    main()