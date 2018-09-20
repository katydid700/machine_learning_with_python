"""
author : louis
date : 09/19/2018
function ; classification of flowers
version : 1.0
"""
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
def main():
    """
    主函数
    """
    pass

    iris_dataset = load_iris()
    print('keys of iris_dataset: \n{}'.format(iris_dataset.keys()))

    # #targer_names键对应的值是一个字符串数组,里面包括了我们要预测的花的品种
    # print('target names :{}'.format(iris_dataset['target_names']))
    # #feature_names 键对应的值是一共字符串列表, 对每一个特征进行了说明:
    # #sepal : 萼片 , petal : 花瓣
    # print('feature names :{}'.format(iris_dataset['feature_names']))
    # #data数组的每一行对应一朵花,列代表每朵花的四个测量数据
    # print('shape of data:{}'.format(iris_dataset['data'].shape))
    # print('前五多花的数据:\n{}'.format(iris_dataset['data'][:5]))
    # #品种被转换成0-2的整数:(其中 0-setosa , 1-versicolor , 2-virginica
    # print('Target:\n{}'.format(iris_dataset['target']))


    #约定训练数据
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'], random_state=0)
    # print('X_train shape: {}'.format(X_train.shape))
    # print('y_train shape: {}'.format(y_train.shape))
    #
    # print('X_test shape: {}'.format(X_test.shape))
    # print('y_test shape: {}'.format(y_test.shape))

    #k近邻算法
    knn = KNeighborsClassifier(n_neighbors=1)
    # fit 方法,基于训练集来构建模型
    print(knn.fit(X_train, y_train))

    #下面有一朵新花
    X_new = np.array([[5, 2.9, 1, 0.2]])
    print('X_new.shape :{}'.format(X_new.shape))

    #调用knn对象的predict 方法来进行预测
    prediction = knn.predict(X_new)
    print('prediction:{}'.format(prediction))
    print('predicted target name :{}'.format(
        iris_dataset['target_names'][prediction]
    ))

    #计算精度(accuracy)离衡量模型优劣
    y_pred = knn.predict(X_test)
    print('Test set predictions:\n{}'.format(y_pred))

    print('test set scroe :{}'.format(np.mean(y_pred == y_test)))




if __name__ == '__main__':
    main()