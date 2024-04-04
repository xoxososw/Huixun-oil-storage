# -*- coding: utf-8 -*-
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
import matplotlib.pylab as plt
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn import tree
from scipy import *
from sklearn import preprocessing
import matplotlib as mpl
import math
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score,accuracy_score
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import lgis,Random_ForestCX,svm,gbdtCX,xgbstCX,adaboostCX
import lgis,Random_ForestSJHF,svm,gbdtSJHF,xgbstSJHF,adaboostSJHF
gbk="gb2312"
utf='utf_8_sig'
# 计算不对称的KL散度
def asymmetricKL(P, Q):
    print(sum(P * np.log(P / Q)))
    return sum(P * np.log(P / Q))  # calculate the kl divergence between P and Q

#计算对称的KL散度
def symmetricalKL(P, Q):
    return ((asymmetricKL(P, Q) + asymmetricKL(Q, P))/ 2.00)

def caclutor(P,Q,varlist):#计算pvalue的均值
    num=0
    count=0
    for m in varlist: #遍历每个参数
        res=ks_2samp(P[m], Q[m])
        count=count+res[1]
        num=num+1
    return  count/num

def draw(P,Q,varlist,model): #绘制各参数的散度图像
    for m in varlist:
        plt.figure(figsize=(12, 9))
        ax1 = sns.kdeplot(P[m], label='train_'+str(m))
        ax2 = sns.kdeplot(Q[m], label='test_'+str(m))
        plt.title(m, size=10)
        plt.show()
        # plt.savefig('../data/img/'+model+'_'+m+'.png')

def random_part(data,model):
    y = data['y_label']
    # X = data.drop(['cellid', 'y_label','DEPTH'], axis=1)
    varlist=data.columns.tolist()
    # print(type(varlist))
    varlist.remove('cellid')
    varlist.remove('y_label')
    max=0
    for i in range(100):
        # 按照7:3划分训练数据与测试数据###
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3, stratify=y) #划分100次
        df_train=X_train[varlist]
        df_test=X_test[varlist]
        res=caclutor(df_train,df_test,varlist)
        if (res>=max):  #保存pvalue均值最大的划分结果
            max=res
            print('已进行一次划分')
            #print("x_test",X_test)
            X_test.to_csv('./result/' + model + '/' + model + '_ce.csv', encoding=utf, index=False)
            X_train.to_csv('./result/' + model + '/'+ model + '_xun.csv', encoding=utf, index=False)
    print('划分结束_',max)


def SJHF():
    #改
    # 要保存到各个模型output文件夹下的路径
    # str_col = "pattern_5RA25"
    # str_col = "6RA25"
    str_col = "6RT"
    path1 = f'./特征提取/pca/小层+测井数据+录井+训练+{str_col}-addShui.csv'
    # # 读取训练集提取的特征数据
    f = open(path1, encoding=utf)
    data = pd.read_csv(f)
    data = data.fillna(-1)
    data = data.drop_duplicates()
    print(data.shape)
    #
    #改
    # str_col = "四个模型"
    if (os.path.exists('./result/' + str_col + '/'+str_col+'_ce.csv') == False):
        random_part(data, str_col)  # 通过此函数将数据集分为训练数据与测试数据
    else:
        print('数据已存在')
    # data_wator = data[data['y_label'] == 0]
    # print(data_wator.shape)
    # data_wator = data_wator[:55]
    # data_oil = data[data['y_label'] == 1]
    # data_oil = data_oil[:data_wator.shape[0]]
    # print(data_oil.shape)
    # print(data_oil.shape)
    # data = pd.concat([data_oil,data_wator])
    # data = pd.concat([data,data_wator])
    # print(data.shape)
    X_train = pd.read_csv(f"./result/{str_col}/{str_col}_xun.csv", encoding='utf-8')
    X_train = X_train.fillna(-1)
    X_train = X_train.drop_duplicates()
    ##获取参数的列名##
    varlist = X_train.columns.tolist()
    varlist.remove('cellid')
    varlist.remove('y_label')
    ##绘制图像##
    # df_train=X_train[varlist]
    # df_test= pd.read_csv('../data/' + str_col + '_测试集.csv', encoding=utf)
    # df_test=df_test[varlist]
    # draw(df_train,df_test,varlist,str_col)
    y_train = X_train['y_label']
    X_train = X_train.drop(['cellid', 'y_label',], axis=1)
    # 三个模型
    # adaboostSJHF.main(X_train,y_train,str_col)
    xgbstSJHF.main(X_train,y_train,str_col)
    # gbdtSJHF.main(X_train,y_train,str_col)
    # Random_ForestSJHF.main(X_train,y_train,str_col)
    # mlp_class2.main(X_train,y_train,str_col)

def CX():
    path1 = ".\特征提取\pca\小层+测井数据+录井+训练+5RA25-addShui.csv"
    path2 = ".\特征提取\pca\小层+测井数据+录井+测试+5RA25.csv"
    # 读取训练集提取的特征数据
    f = open(path1, encoding="utf-8")
    data = pd.read_csv(f, )
    data = data.fillna(-1)
    print(data.shape)
    f = open(path2, encoding="utf-8")
    data1 = pd.read_csv(f, )
    data1 = data1.fillna(-1)
    print(data1.shape)
    y = data['y_label']
    X = data.drop(['cellid', 'y_label'], axis=1)
    pre_y = data1['y_label']
    pre = data1.drop(['cellid', 'y_label'], axis=1)
    # print(pre.shape)
    # 要保存到各个模型output文件夹下的路径
    str_col = "5RA25"
    # str_col = "6RA25_0302"
    # str_col = "6RT_0302"
    # 三个模型
    adaboostCX.main(X, y, str_col, pre, pre_y)
    # xgbstCX.main(X,y,str_col,pre,pre_y)
    # gbdtCX.main(X,y,str_col,pre,pre_y)
    # Random_ForestCX.main(X,y,str_col,pre,pre_y)

def CX_zidingyin():
    path1 =".\特征提取\pca\小层+测井数据+录井+训练+5RA25-addShui.csv"
    # 读取训练集提取的特征数据
    f = open(path1, encoding="utf-8")
    data = pd.read_csv(f, )
    data = data.fillna(-1)
    print(data.shape)

    oilCeng = data[data['y_label'] == 1]
    print("油层：",oilCeng.shape)
    shuiCeng = data[data['y_label'] == 0]
    print("水层：",shuiCeng.shape)
    # 要划分的水层/油层的个数
    numCeXun = int(data.shape[0]*0.2/2)

    # 拿到固定的油层
    cengGuding = ["港5-66-3$1439.9","港5-66-3$1436.5","港3-55-2$1387.3","港3-55-2$1382.4","港3-55-2$1381.7","港3-55-2$1378.8",
    "港3-55-2$1373.4","港9-25-3$1831.1","港9-25-3$1826.2","港9-25-3$1826.2","港9-25-3$1831.1"]



    # 随机划分50次
    for i in range(1):
        youCe = pd.DataFrame()
        for i in range(len(cengGuding)):
            id = cengGuding[i]
            tmp = oilCeng[oilCeng['cellid'] == id]
            oilCeng = oilCeng[oilCeng['cellid'] != id]
            youCe = pd.concat([youCe, tmp])
        restOil = numCeXun - youCe.shape[0]

        # 测试集的油和水
        youCeLast = oilCeng.sample(n=restOil)
        youCe = pd.concat([youCe, youCeLast])
        shuiCe = shuiCeng.sample(n=numCeXun)
        data1 = pd.concat([youCe,shuiCe])

        # 训练集上去掉测试集的油和水
        ceCellid = data1['cellid'].values.tolist()
        dataXun = pd.concat([oilCeng,shuiCeng])
        for ii in range(len(ceCellid)):
            id = ceCellid[ii]
            dataXun = dataXun[dataXun['cellid'] != id]
        print("训练集：",dataXun.shape)
        print("训练集油：",dataXun[dataXun['y_label']==1].shape)
        print("训练集水：",dataXun[dataXun['y_label']==0].shape)
        print("测试集：",data1.shape)
        print("测试集油：", data1[data1['y_label'] == 1].shape)
        print("测试集水：", data1[data1['y_label'] == 0].shape)
        y = dataXun['y_label']
        X = dataXun.drop(['cellid', 'y_label'], axis=1)
        pre_y = data1['y_label']
        pre = data1.drop(['cellid', 'y_label'], axis=1)
        # 要保存到各个模型output文件夹下的路径
        str_col = "5RA25"
        # str_col = "6RA25_0728"
        # str_col = "6RT"
        # 三个模型
        #
        adaboostCX.main(X, y, str_col, pre, pre_y)
        # xgbstCX.main(X,y,str_col,pre,pre_y)
        # gbdtCX.main(X,y,str_col,pre,pre_y)

        # Random_ForestCX.main(X,y,str_col,pre,pre_y)


if __name__=='__main__':   #主程序最顶端，加入此行代码即可

    SJHF()

    # CX()
    # CX_zidingyin()


