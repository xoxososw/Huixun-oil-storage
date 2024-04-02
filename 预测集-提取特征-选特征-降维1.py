# time:2021/5/2 低阻提取特征
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from scipy import *
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from tsfresh import extract_features, extract_relevant_features, select_features
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import OrderedDict
mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings(action='ignore')


# print(dataRT1.shape[0])
def feature_extraction(dataRT,col):#提取特征
    dfRT = dataRT[col]
    dfRT = dfRT.dropna()
    print(dfRT.shape)
    cellid = dfRT['cellid'].drop_duplicates().values.tolist()
    print(len(cellid))
    X_RT=extract_features(dfRT, column_id='cellid', column_sort='depth', n_jobs=4)
    print(X_RT.shape[0])
    X_RT['cellid'] = cellid
    dataRTY = dataRT[['cellid','y_lable']].drop_duplicates()
    X_RT = X_RT.merge(dataRTY,on='cellid',how='left')
    print(X_RT.shape)
    return X_RT


def RF(name):#随机森林挑选top50特征
    datax=pd.read_csv(name,encoding='utf_8_sig')
    #data_x=data_x.drop(columns=['cellid','生产标签'])
    #data_x=data_x.dropna()
    datax = datax.replace([np.inf, -np.inf], np.nan)
    #data_x = data_x.dropna()
    print(datax.shape)
    data_x=datax.dropna(axis=1,how='any')
    temp_data=data_x
    y_label=data_x['y_lable']
    data_x = data_x.drop(columns=['cellid','y_lable'])
    columns=data_x.columns.tolist()
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    forest.fit(data_x,y_label)
    feature_imp=forest.feature_importances_
    result=pd.DataFrame()
    result['tz']=columns
    result['zycd']=feature_imp
    result=result.sort_values(ascending=False,by=['zycd'])
    feature_list=result['tz'].values.tolist()[:50]
    feature_list.append('cellid')
    feature_list.append('y_lable')
    res=temp_data[feature_list]
    res.to_csv("./特征提取/rf/"+name.split('/')[2],index=False,encoding='utf_8_sig')

def pca(name):#降维并融合录井特征

    f = open(name, encoding='utf-8')
    data = pd.read_csv(f)
    cellid = data["cellid"]
    y_label = data["y_lable"]
    for i in [15]:
        data_pred = data.drop(columns=["cellid", "y_lable"])

        # n_components是降低的维数
        pca = PCA(n_components=i)
        data_pred = pca.fit_transform(data_pred)

        data_pred = pd.DataFrame(data_pred)
        data_pred["cellid"] = cellid
        data_pred["y_label"] = y_label

        data_pred = data_pred.replace("油层","1")
        data_pred = data_pred.replace("水层","0")
        data_pred.to_csv('.\特征提取\pca\\'+name.split('/')[2],index=False,encoding='utf_8_sig')
    return data_pred
def por_transfor(df):
    df['rock'] = df['rock'].replace('粉砂岩', 0)
    df['rock'] = df['rock'].replace('含砾不等粒砂岩', 1)
    df['rock'] = df['rock'].replace('砾状砂岩', 2)
    df['rock'] = df['rock'].replace('泥岩', 3)
    df['rock'] = df['rock'].replace('泥质粉砂岩', 4)
    df['rock'] = df['rock'].replace('泥质砂岩', 5)
    df['rock'] = df['rock'].replace('砂质泥岩', 6)
    df['rock'] = df['rock'].replace('细砂岩', 7)
    df['rock'] = df['rock'].replace('中砂岩', 8)
    df['rock'] = df['rock'].replace('', -1)

    df['ygys'] = df['ygys'].replace('褐色', 0)
    df['ygys'] = df['ygys'].replace('灰白色', 1)
    df['ygys'] = df['ygys'].replace('灰褐色', 2)
    df['ygys'] = df['ygys'].replace('灰黄色', 3)
    df['ygys'] = df['ygys'].replace('灰绿色', 4)
    df['ygys'] = df['ygys'].replace('灰色', 5)
    df['ygys'] = df['ygys'].replace('浅灰色', 6)
    df['ygys'] = df['ygys'].replace('浅棕红色', 7)
    df['ygys'] = df['ygys'].replace('棕红色', 8)
    df['ygys'] = df['ygys'].replace('棕黄色', 9)
    df['ygys'] = df['ygys'].replace('暗黄', 10)
    df['ygys'] = df['ygys'].replace('暗黄色', 10)
    df['ygys'] = df['ygys'].replace('淡黄', 11)
    df['ygys'] = df['ygys'].replace('淡黄色', 11)
    df['ygys'] = df['ygys'].replace('黄', 12)
    df['ygys'] = df['ygys'].replace('黄色', 12)
    df['ygys'] = df['ygys'].replace('黄白', 13)
    df['ygys'] = df['ygys'].replace('亮黄', 14)
    df['ygys'] = df['ygys'].replace('棕黄', 14)
    df['ygys'] = df['ygys'].replace('无', 15)
    df['ygys'] = df['ygys'].replace('', -1)

    df['hyjb'] = df['hyjb'].replace('荧光', 0)
    df['hyjb'] = df['hyjb'].replace('油斑', 1)
    df['hyjb'] = df['hyjb'].replace('油迹', 2)
    df['hyjb'] = df['hyjb'].replace('油浸', 3)
    df['hyjb'] = df['hyjb'].replace('饱含油', 4)
    df['hyjb'] = df['hyjb'].replace('富含油', 5)
    df['hyjb'] = df['hyjb'].replace('', -1)
    return df
if __name__ == '__main__':

    col = [['cellid', 'depth', "ac",  "gr", "sp", "ra25"],
           ['cellid', 'depth', "ac",   "gr", "sp", "ra25"],
           ['cellid', 'depth', "ac",   "gr", "sp", "rt"]
           ]
    name = ['./数据预处理后的训练集和测试集/小层+测井数据+录井+测试+5RA25.csv' ,
            './数据预处理后的训练集和测试集/小层+测井数据+录井+测试+6RA25.csv',
            './数据预处理后的训练集和测试集/小层+测井数据+录井+测试+6RT.csv']

    for i in range(3):
        f = open(name[i], encoding='utf-8')
        dataRT1 = pd.read_csv(f, encoding='utf-8')

        tmp = feature_extraction(dataRT1, col[i])
        tmp.to_csv('./特征提取/tsfresh/'+name[i].split('/')[2],encoding='utf_8_sig', index=False)
        RF('./特征提取/tsfresh/'+name[i].split('/')[2])
        X_feature=pca('./特征提取/rf/'+name[i].split('/')[2])
        layer=pd.read_csv(name[i])
    # layer = pd.DataFrame(list(train_raw_data.objects.values()))
        layer=layer[['cellid','por','hyjb','ygys','rock']]
        layer=layer.drop_duplicates(subset='cellid')
        layer=por_transfor(layer)
        db_feature=pd.merge(left=X_feature,right=layer,left_on='cellid',right_on='cellid',how='left')
        db_feature.to_csv("./特征提取/pca/"+name[i].split('/')[2], encoding='utf-8', index=False)