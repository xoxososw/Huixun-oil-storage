#首先第一步小层数据确定油层打标签
#第二步选取同分布的待预测小层数据
#第三步训练数据和测试数据与测井数据结合
#训练数据和测试数据小层+测井与录井结合
#第四步选参数
#第五步确定油水比
#第六步增加水层
# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score
# Memory management
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc
import numpy as np
import pandas as pd
import numpy as np
import os
import shutil
from pandas import DataFrame, Series

gc.enable()
# Plot
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Suppress warnings
from sklearn.utils import shuffle
from random import choice
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from lightgbm import log_evaluation, early_stopping

#从小层数据中划分训练集和待预测小层并为训练集打标签所有函数
def jsdataprocessing(x):
    #对小层数据的解释序号进行预处理
    d = dict(zip("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sup|Oct|Nov|Dec".split("|"),[str(i) for i in range(1,13)]))
    import re
    s = None
    p1 = re.match('(\d+)月(\d+)日',x)
    p2 = re.match('(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sup|Oct|Nov|Dec)-(\d+)',x)
    p3 = re.match('(\d+)-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sup|Oct|Nov|Dec)', x)
    if p1:
        # 如果A月B日，B日小于12，则要转成A-B，如6月1日转成6-1
        # 如果A月B日，B日大于12，则要转成B-A，如1月12日转成12-1
        if float(p1.groups()[1])>12:
            s = str(p1.groups()[1]+"-"+p1.groups()[0])
        else:
            s = str(p1.groups()[0] + "-" + p1.groups()[1])
        # s = 'a'
    if p2:
        s = str(p2.groups()[1]+"-"+d[p2.groups()[0]])
        # s = 'a'
    if p3:
        s = str(d[p3.groups()[1]]+ "-" +p3.groups()[0])
        # s = 'a'
    if s is None:
        s = x
    return s
def skoilceng(xcdata):
    #找到已经射孔小层,并对解释序号进行处理
    # data=pd.read_csv(readxcpath,encoding="utf_8")
    xcdata['解释序号'] = xcdata['解释序号'].apply(lambda x: str(x))
    xcdata['解释序号'] = xcdata['解释序号'].apply(lambda x: jsdataprocessing(x))
    xcdata['解释序号']="解释序号:"+xcdata['解释序号']
    xcdata1 = xcdata[xcdata["射孔情况"] == "是"]
    xcdata1 = xcdata1.reset_index(drop=True)
    xcdata1['cellid'] = xcdata1['井号'].str.cat(xcdata1['砂层顶深'].astype(str), sep='$')
    xcdata2 = xcdata[xcdata["射孔情况"] == "否"]
    xcdata2 = xcdata2.reset_index(drop=True)
    xcdata2['cellid'] = xcdata2['井号'].str.cat(xcdata2['砂层顶深'].astype(str), sep='$')
    return xcdata1,xcdata2
    # print(data)
    # xcdata.to_csv(savexcpath,encoding="utf_8",index=False)
def dizuoilceng(xcdata,scdata):
    print("xcdata",xcdata)
    print("scdata", scdata)
    #找到低阻油层
    #思路：通过小层数据的井号+解释序号对应生产数据的井号+生产序号，对初始生产日产油量进行批分，判断是否小于三吨，结合含水判断是否为油层
    # oilceng=pd.DataFrame()
    xcdata["y_label"]=""
    for i in range(len(xcdata)):
        # print(xcdata)
        # print(xcdata.loc[1, '井号'])
        jingname=xcdata.loc[i,'井号']
        if '电测解释结论' in list(xcdata.columns):
            jielun = xcdata.loc[i,'电测解释结论']
        if "解释结论描述" in list(xcdata.columns):
            jielun=xcdata.loc[i,'解释结论描述']

        # print("井名",jingname)
        # print(xcdata.loc[i,'解释序号'])
        # print("电测解释结论",jielun)
        jsxuhao=xcdata.loc[i,'解释序号'].split(":")[1]
        # print("解释序号",xcdata.loc[i,'解释序号'])
        scdata1=scdata[(scdata["井号"]==jingname)&(scdata["生产层号"].str.contains(jsxuhao))]
        if len(scdata1)>0:
            scdata1=scdata1.sort_values(by="年月",ascending=True)
            scdata1=scdata1.reset_index(drop=True)
            # print(scdata1)
            num=len(scdata1.loc[0,'生产层号'].split(","))
            oilliang=scdata1.loc[0,'日产油量']
            waterliang = scdata1.loc[0, '含水']
            if (oilliang/num)>3 or waterliang<0.99:
                # oilceng=oilceng.append(xcdata.loc[i])
                xcdata.loc[i,"y_label"]="油层"
            if (waterliang>0.99) and (jielun=="水层"):
                print("解释结论描述",jielun)
                xcdata.loc[i, "y_label"] = "水层"
    xcdata=xcdata[(xcdata["y_label"]=="水层")|(xcdata["y_label"]=="油层")]
    return xcdata
#同分布的所有函数
def dataprocess(df_train,df_test):
    #对数据进行预处理
    '''
    df_train为带预测小层数据
    df_test为已核实的小层数据
    '''
    print('函数开始测试集shape', len(df_test))
    # df_train = df_train[['泥质含量', '孔隙度', '含油饱和度', '砂层厚度', 'cellid']]  # 选取特征
    df_train = df_train.dropna(axis=0, subset=['泥质含量', '孔隙度', '含油饱和度', '砂层厚度', 'cellid'])  # 将没有这些参数的数据删除
    df_train.reset_index(drop=True)  # 删除完重置索引
    df_train=df_train.drop_duplicates(subset=['cellid'])
    df_train=df_train.reset_index(drop=True)
    df_train_len = len(df_train)
    # df_test = df_test[['泥质含量', '孔隙度', '含油饱和度', '砂层厚度', 'cellid','电测解释结论']]  # 测试集中选取参数多一个解释结论标签
    df_test = df_test.dropna(axis=0, subset=['泥质含量', '孔隙度', '含油饱和度', '砂层厚度', 'cellid','电测解释结论'])
    df_test.reset_index(drop=True)
    # print(df_train.columns)
    print('参数测试集shape', len(df_test))
    df_test = df_test.drop(columns='电测解释结论')
    df_test = df_test.drop_duplicates(subset=['cellid'])
    print('去重测试集shape', len(df_test))
    df_test = df_test.reset_index(drop=True)
    # df_test_len = len(df_test)
    # columns_select = df_test.columns.values.tolist()  # 参数列表
    print('训练集shape')
    print(df_train.shape)
    print(df_train.columns)
    print('测试集shape')
    print(df_test.shape)
    print(df_test.columns)

    # df_train, df_test, train_index, test_index = ceil_split(df_all,'y_label')
    # print(df_train.columns)
    # print('预测小层保留的50列特征',df_train.columns)
    df_all = pd.concat([df_train, df_test])  # 将训练集和测试集合并
    print(df_all.shape)
    # df_all.to_csv("./同分布所需数据/test.csv",encoding="utf_8",index=False)
    # df_all.reset_index(drop=True)
    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将数据中正无穷和负无穷的数据用np.nan代替
    # df_all = df_all.dropna(subset=['y_label'])
    df_all = df_all.drop_duplicates(subset=['cellid'])
    df_all = df_all.reset_index(drop=True)
    print(df_all.shape)
    # 去掉缺失值太多的变量
    bad_cols = []
    for col in df_all.columns:
        rate_train = df_all[col].value_counts(normalize=True, dropna=False).values[0]
        if rate_train > 0.99:  # 如果col缺失值太多就放到bad_cols，因为col正常数据不可能一样，只可能是数据为np.nan
            bad_cols.append(col)
    # print(len(bad_cols))
    df_all = df_all.drop(bad_cols, axis=1)  # 将数据缺失很厉害的数据删除掉
    print('去掉缺失值后Data Shape: ', df_all.shape)
    # 划分训练集和测试集（这里应该绝对需要划分）
    num=len(df_train)
    # print("df_train数据长度为",num)
    # 不懂 ！！
    df_train = df_all[num:]
    print("df_train.shape",df_train.shape)
    df_test = df_all[:num]
    print("df_test.shape",df_test.shape)
    # 对抗验证
    df_train['Is_Test'] = 1  # 将训练集数据打上标签1这里的训练集应该是已核实的小层数据
    df_test['Is_Test'] = 0  # 带预测的小层数据
    df_test = pd.DataFrame(df_test)
    # df_test.to_csv(r'E:\石油资料\港中分析方法迁移到港东\训练集数据.csv',encoding='utf_8_sig',sep=',')
    # 去掉列名中的特殊字符
    df_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in
                        df_train.columns]  # isalnum()如果是字符或者10进制数字就为真。汉字也是真
    df_test.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df_test.columns]
    # 将 Train 和 Test 合成一个数据集。HasDetections是数据本来的Y，所以剔除。
    df_adv = pd.concat([df_train, df_test])  # 将训练集和训练集合并
    # print(df_adv.columns)
    df_adv = df_adv.reset_index(drop=True)
    df_adv = df_adv.fillna(0)  # 用0将空值填空
    # df_adv = shuffle(df_adv)
    df_adv=df_adv[['泥质含量', '孔隙度', '含油饱和度', '砂层厚度','Is_Test',"cellid"]]
    # df_adv1 = df_adv[['泥质含量', '孔隙度', '含油饱和度', '砂层厚度','Is_Test',"cellid"]]
    df_adv1=df_adv.drop(columns=['Is_Test', 'cellid'])
    scaler = StandardScaler()
    X_number = scaler.fit_transform(df_adv1.astype(float))  # 数据标准化
    return X_number,df_adv,df_train,df_test

def modelbuilding(X_number,df_adv):
    '''
    :param X_number: 标准化之后的数据集（不带Is_Test,cellid）
    :param df_adv: 未标准化的数据集（带Is_Test,cellidw）
    :return: 返回预测的df_adv预测为1的概率
    '''
    # df_adv1=df_adv.drop(columns=['Is_Test', 'cellid'])
    adv_data = lgb.Dataset(
        data=X_number, label=df_adv.loc[:, 'Is_Test'])
    # print(df_adv.columns)
    # 定义模型参数
    params = {
        'boosting_type': 'gbdt', #指定了 boosting 的类型，这里设为 'gbdt'，代表梯度提升决策树。
        'colsample_bytree': 1, #表示在构建每棵树时特征采样的比例，这里设为 1，即不进行特征采样。
        'learning_rate': 0.07, #学习率，控制每次迭代中模型参数更新的幅度
        'max_depth': 3, #每棵树的最大深度限制
        'min_child_samples': 5, #每个叶子节点上最少的样本数量
        # 'min_child_weight': 5,
        'min_split_gain': 0.0, #分裂节点时所需的最小增益
        'num_leaves': 10, #每棵树的叶子节点数
        'objective': 'binary', #指定了优化的目标函数，这里设为 'binary'，代表二分类问题
        'random_state': 50, #随机种子，用于控制模型的随机性，保证结果的可重复性
        'subsample': 0.8, #训练每棵树时用于训练的样本比例
        'metric': 'auc', #评估指标，这里设为 'auc'，
        'num_threads': 8, #用于并行学习的线程数量
    }
    # 2、对抗交叉验证
    adv_cv_results = lgb.cv(
        params,
        adv_data,
        num_boost_round=10000, #指定了最大迭代次数，即最多训练的树的数量
        nfold=7, #指定了交叉验证的折数
        # categorical_feature=categorical_columns,
        callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=200)],  #是一个包含回调函数的列表，用于在训练过程中执行特定的操作。其中包括了 log_evaluation 回调函数用于记录评估指标的值，以及 early_stopping 回调函数用于提前停止训练以避免过拟合
        shuffle=True, #指定是否在每次交叉验证前对数据进行洗牌。
        seed=0)
    # print('测试+训练集交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
    #     adv_cv_results['auc-mean'][-1], adv_cv_results['auc-stdv'][-1]))
    # print('测试+训练集模型最优的迭代次数为{}.'.format(len(adv_cv_results['auc-mean'])))
    # 3、我们使用训练好的模型，对所有的样本进行预测，得到各个样本属于测试集的概率
    params['n_estimators'] = len(adv_cv_results['valid auc-mean'])  # 最优的迭代次数
    model_adv = lgb.LGBMClassifier(**params)
    df_adv_data = df_adv.drop(columns=['Is_Test', 'cellid'])
    df_adv_label = df_adv['Is_Test']
    # print(df_adv_data.head)
    model_adv.fit(df_adv_data, df_adv_label)
    aaa= model_adv.predict_proba(df_adv.drop(columns=['Is_Test', 'cellid']))
    pre= model_adv.predict(df_adv.drop(columns=['Is_Test', 'cellid']))
    # print(pre[1000:1100])
    # print("aaa",aaa[1000:1100])
    preds_adv = model_adv.predict_proba(df_adv.drop(columns=['Is_Test', 'cellid']))[:, 1]#预测为1的概率
    return preds_adv

def selectsamlay(df_train,preds_adv):
    '''
    :param df_train: 待预测小层数据
    :param preds_adv: 待预测小层数据的预测概率
    :return:
    '''
    df_train_copy = df_train.copy()
    print(df_train_copy.shape)
    print(len(preds_adv))  #有多少个样本被用于预测
    df_train_copy['is_test_prob'] = preds_adv[:len(df_train)]  # 预测为是is_test_prob的概率多少
    # print("preds_adv",preds_adv[len(df_train):])
    #    根据概率排序
    df_train_copy = df_train_copy.sort_values('is_test_prob').reset_index(drop=True)
    #           将为1概率最大的20%作为验证集，
    # print("is_test_prob",df_train_copy['is_test_prob'])
    # df_train_copy.to_csv("./待预测小层数据/test.csv")
    df_validation_2 = df_train_copy.iloc[int(0.8 * len(df_train)):, ]
    df_train_2 = df_train_copy.iloc[:int(0.8 * len(df_train)), ]
    # print("df_train_2",df_train_2)
    print('训练集样本个数为', df_train_2.shape)
    print('验证集样本个数为', df_validation_2.shape)
    # print(df_validation_2)
    df_validation_2 = pd.DataFrame(df_validation_2)
    # df_validation_2.to_csv(r'E:\石油资料\港中分析方法迁移到港东\方法优化\分布不相似预测集.csv',encoding='utf_8_sig',sep=',')
    # df_validation_2 = df_validation_2.drop(columns=['is_test_prob'])
    # df_train_2 = df_train_2.drop(columns=['is_test_prob'])
    return df_validation_2
# 测井提取及测井与训练集测试集融合
def txt2pd(fpath,outpath):
    # 将测井的txt文件转化为csv文件
    # shutil.rmtree(outpath)
    files = os.listdir(fpath)
    for f1 in files:
        temp_f1=os.path.join(fpath,f1)
        temp_f2=os.listdir(temp_f1)
        for f2 in temp_f2:
            fname=f2.split('.')[0]+'.csv'
            path=os.path.join(temp_f1,f2)
            read = False
            temp_global = ''
            temp2 = []
            write = False
            nmark = 0
            all = ''
            columns=''
            try:
                with open(path,'r',encoding="utf_8") as f:
                    # print('read')
                    temp = ''
                    for line in f:
                        if '#DEPTH' in line:
                            # print('找到列名 总是')
                            # print(line)
                            # print('需要的是')
                            columns = line.split()
                            columns[0] = 'DEPTH'
                            # print("column is ")
                            # print(columns)
                            read = True
                        if read:
                            temp += line
                        nmark = line
                    temp_global = temp
            except:
                read = False
                temp = ''
                print('temp',temp)
                with open(path,'r',encoding="gbk") as f:
                    # print('read')
                    for line in f:
                        if '#DEPTH' in line:
                            # print('找到列名 总是')
                            # print(line)
                            # print('需要的是')
                            columns = line.split()
                            columns[0] = 'DEPTH'
                            # print("column is ")
                            # print(columns)
                            read = True
                        if read:
                            temp += line
                        nmark = line
                    temp_global =temp

            temp = temp_global.split('\n')[1:]
            # print(len(temp))
            # print(temp[1:10])
            count = 0
            for line in temp:
                count += 1
                j = line.split()
                if(len(j) == 36):
                    print("error")
                    print(count)
                    print(j)
                t = []
                if j!=[]:
                    for k in j:
                       try:
                            k = float(k)
                       except:
                           k=0.000
                       t.append(k)
                    temp2.append(t)

            df = pd.DataFrame(data=temp2,columns=list(columns))

            try:
                df = pd.read_csv(path,encoding="utf_8_sig", sep='\s+')
            except:
                df = pd.read_csv(path, encoding="gbk", sep='\s+')

            df.rename(columns={"#DEPTH": "DEPTH"}, inplace=True)
            o_path = os.path.join(outpath, f1 + '.csv')
            df.to_csv(o_path, encoding='utf_8_sig', sep='，', index=False)
            # o_t=os.path.join(outpath,f1)#输出路径+子目录
            # print(f1)
            # if os.path.exists(o_t):
            #     o_path=os.path.join(o_t,fname)#输出路径+子目录+文件名称
            # else:
            #     os.mkdir(o_t)#如果没有子文件夹则先行创建
            #     o_path = os.path.join(o_t, fname)  # 输出路径+子目录+文件名称
            # df.to_csv(o_path, encoding='utf_8_sig',sep=',',index=False)
def extract(path):
    # 从指定路径下读取txt
    file = open(path)

    cnt = 0
    col = []
    df = []

    while True:
        text = file.readline()  # 只读取一行内容
        cnt += 1
        if cnt < 5 or cnt == 6 or cnt == 7:
            continue

        # 判断是否读取到内容
        if not text:
            break
        # print(text)
        text = text[:-1]

        # 处理字段
        if cnt == 5:
            # print(text, end="")
            text = text[13:]
            text = text.split(",")
            col.append('DEPTH_index')

            col.append(text[0])
            for i in text[1:-1]:
                col.append(i[1:])
            col.append(text[-1].split('\\')[0])
            # print(col)
            if col[-1] == col[-2]:
                col.pop()
            # print(col)
        if cnt >= 8:
            val = []
            res = text.split(' ')
            for v in res:
                if v != '':
                    val.append(v)
            df.append(val)
    file.close()

    return pd.DataFrame(df,columns=col)
def findName(f2):
    # 指定要读取的文件名称
    ans = []
    # for f in f2:
        # if '拼接' in f:
        #     if '标准' in f:
        #         ans.append(f)
        #         return ans
        #     if '综合' in f:
        #         ans.append(f)
        #         return ans
    for f in f2:
        if '标准' in f and '第一次完井' in f:
            ans.append(f)
            # return ans

    for f in f2:
        if '标准' in f and '第二次完井' in f:
            ans.append(f)
            # return ans

    for f in f2:
        if '标准' in f:
            ans.append(f)
            # return ans
    return ans
def txt2merge_csv(test_data,out_path):
    '''
    将测井txt合并为1个csv文件
    :param test_data: 测井txt所在的总文件夹，下设各井单独的文件夹，再下是每个井的多个测井txt
    :param out_path: 结果的输出文件夹
    :return:
    '''
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        os.mkdir(out_path)
    else:
        os.mkdir(out_path)
    for root, dirs, files in os.walk(test_data):
        # print("dirs",dirs)
        for dir in dirs:
            # print(root+'/'+dir)
            res = pd.DataFrame()
            for r2, d2, f2 in os.walk(root + '\\' + dir):
                # print("路径",root + '/' + dir)
                nameList = findName(f2)
                cnt = 0
                for f in nameList:
                    cnt += 1
                    # 提取文本数据
                    data = extract(root + '\\' + dir + "\\" + f)
                    data['DEPTH_index'] = data['DEPTH_index'].astype('float')
                    # 把井号加进去
                    well = []
                    name = f.split('@')[0]
                    for w in range(data.shape[0]):
                        well.append(name)
                    data['井号'] = name
                    if cnt == 1:#如果该井文件夹下只有一个txt文件
                        res = data
                    else: #如果该井文件夹下有多个txt文件
                        # print(res.columns)
                        if data['DEPTH_index'].min() < res['DEPTH_index'].min():
                            res = res[res['DEPTH_index'] > data['DEPTH_index'].max()]
                        else:
                            data = data[data['DEPTH_index'] > res['DEPTH_index'].max()]
                        res = pd.concat([res, data])

                    del data
            if res.shape[0] == 0:
                continue

            res = res.sort_values(by=['DEPTH_index'])
            res = res.drop_duplicates(subset=['DEPTH_index'], keep='first')
            print("文件名和文件中的参数",f,res.columns)
            res.to_csv(out_path + dir + "@测井合并.csv", encoding='utf_8_sig', index=False)

def select_test_data(test_f,layer_f,out_f):
    if os.path.exists(out_f):
        shutil.rmtree(out_f)
        os.mkdir(out_f)
    else:
        os.mkdir(out_f)
    #根据小层号与顶深底深，从测井csv中选择出对应的测井数据
    # 设定path为港东测井数据所在的文件夹
    path = test_f
    # 遍历path下的各文件夹
    files = os.listdir(path)
    count = 0
    list1 = []
    cejing_big = pd.DataFrame()
    # 读取小层文件
    try:
        low_data = pd.read_csv(layer_f, encoding='utf_8_sig')
    except:
        low_data = pd.read_csv(layer_f, encoding='gbk')
    #筛选未射孔的小层
    # low_data = low_data[low_data['射孔情况'] == '否']
    # 给每一个小层文件设置cellid和id
    print('小层数据大小0', low_data.shape)
    low_data['cellid'] = low_data['井号'].str.cat(low_data['砂层顶深'].astype(str), sep='$')
    low_data['id'] = low_data['井号'].str.cat(low_data['小层号'].astype(str), sep='$')
    print('小层数据大小', low_data.shape)
    # low_data = low_data[low_data['射孔情况'] == '是']
    # low_data = low_data[low_data['电测解释结论']=='水层']
    # print(low_data.columns.values)

    # 遍历测井数据文件找到各小层测井数据
    for f in files:
        path1 = os.path.join(path, f)
        print('当前文件', f)
        # 到达井文件，提取井号
        temp = f.split('.')
        # temp = f.split('.')
        print('当前井号', temp[0])
        # print('小层数据中有的井号',xiaoceng_data['井号'].values)
        # 如果井号能对应上就筛选出该井所有小层的测井数据
        if temp[0] in low_data['井号'].values:
            print('对应上的', temp[0])
            data = pd.DataFrame()
            cejing_data = pd.read_csv(path1, encoding='utf_8_sig', sep=',')
            cejing_data = cejing_data.rename(columns={'DEPTH': '#DEPTH'})
            cejing_data = cejing_data.rename(columns={'深度': '#DEPTH'})
            print(cejing_data.columns)
            # 筛选出小层数据中对应井名的小层数据
            low_data1 = low_data[temp[0] == low_data['井号']]
            # 遍历小层数据中对应当前测井数据井的dataframe，（一行一行读取）
            for index1, row1 in low_data1.iterrows():  #index1表示当前行的索引，row1则表示当前行的数据，通常以Series的形式呈现。
                # 对于小层数据中的每一层，在测井数据中筛选出该层对应的测井数据
                # 第一步，筛选出测井数据的（Depth<当前行的砂层顶深）的所有测井数据，生成一个dataframe
                cejing_temp = cejing_data.loc[row1['砂层顶深'] <= cejing_data['#DEPTH']]
                # 第二步，筛选出测井数据的（Depth>当前行的砂层底深）的所有测井数据，生成一个新的dataframe
                # 最后也就得到了几条这个层里的测井数据
                cejing_temp2 = cejing_temp.loc[cejing_temp['#DEPTH'] <= row1['砂层底深']]
                print("cejing_temp1", cejing_temp2.shape)
                column_list = cejing_temp2.columns.values
                print(column_list)
                cejing_temp1 = cejing_temp2
                # RA25 = pd.DataFrame()
                # 不管是否低阻，将带有RA25的数据筛出
                # 判断低阻层结束
                # 在新的测井数据的dataframe中添加上解释结论描述等列并赋值，同一层理所当然有相同的值
                if not (cejing_temp1.empty):
                    # 说明此小层有对应的测井数据，有对应测井数据的小层又增加了一个
                    count += 1
                else:
                    # 记录下虽然有测井数据对应的井但是没有对应小层的层号
                    print('该井测井数据无对应的小层', temp[0])
                    # count1 +=1
                    list1.append(temp[0])
                    continue
                    # print("cejing_temp1",cejing_temp1)
                if not cejing_temp1.empty:
                    # cejing_temp1['砂层顶深'] = row1['砂层顶深']
                    # cejing_temp1['砂层底深'] = row1['砂层底深']
                    # cejing_temp1['砂层厚度'] = row1['砂层厚度']
                    # cejing_temp1['解释日期'] = row1['解释日期']
                    # cejing_temp1['解释序号'] = row1['解释序号']
                    # cejing_temp1['射孔情况'] = row1['射孔情况']
                    # cejing_temp1['解释结论描述'] = row1['解释结论描述']
                    # cejing_temp1['解释结论标签'] = row1['解释结论标签']
                    # cejing_temp1['油层组'] = row1['油层组']
                    # cejing_temp1['同深度'] = row1['同深度']
                    cejing_temp1['cellid'] = row1['cellid']
                    cejing_temp1['id'] = row1['id']
                    # cejing_temp1['pro_index'] = row1['pro_index']
                    # cejing_temp1['射孔日期'] = row1['射孔日期']
                    # cejing_temp1['射孔段顶深'] = row1['射孔段顶深']
                    # cejing_temp1['射孔段底深'] = row1['射孔段底深']
                    # cejing_temp1['月产水量'] = row1['月产水量']
                    # cejing_temp1['月产气量'] = row1['月产气量']
                    # cejing_temp1['月产油量'] = row1['月产油量']
                    # cejing_temp1['生产层号'] = row1['生产层号']
                    # cejing_temp1['累积产气量'] = row1['累积产气量']
                    # cejing_temp1['累积产水量'] = row1['累积产水量']
                    # cejing_temp1['累积产油量'] = row1['累积产油量']
                    # cejing_temp1['lu_index'] = row1['lu_index']
                    # cejing_temp1['含油级别'] = row1['含油级别']
                    # cejing_temp1['岩石颜色'] = row1['岩石颜色']
                    # cejing_temp1['岩石名称'] = row1['岩石名称']
                    # cejing_temp1['颜色号'] = row1['颜色号']
                    # cejing_temp1['岩性描述'] = row1['岩性描述']
                    # cejing_temp1['顶界井深'] = row1['顶界井深']
                    # cejing_temp1['底界井深'] = row1['底界井深']
                    # cejing_temp1['录井测井解释'] = row1['录井测井解释']
                    # cejing_temp1['孔隙度'] = row1['孔隙度']
                    # cejing_temp1['泥质含量'] = row1['泥质含量']
                    # cejing_temp1['潜力结果'] = row1['潜力结果']
                    # cejing_temp1['井号'] = row1['井号']
                    # cejing_temp1['泥浆电阻率'] = row1['泥浆电阻率']
                    data = pd.concat([cejing_temp1, data])
                # 将筛选过的dataframe给concat给data这个大的数据表
            # 一个井一个数据表，所以就个这口井来添加上一列井号
            if not data.empty:
                data['井号'] = temp[0]
                cejing_big = pd.concat([cejing_big, data])
                print(temp[0] + '这个井的层数', len(list(data['cellid'].unique())))
                data.to_csv(out_f + temp[0] + '@test.csv',
                            index=False, header=True, encoding='utf_8_sig')
        else:
            pass

    print('有测井数据的小层数目', count)
    # print(count1)
    # data['井测没录'] = list
    # data1['层录没测'] = list1


def concat_test(test_f,out_f,out_name):
    if os.path.exists(out_f):
        shutil.rmtree(out_f)
        os.mkdir(out_f)
    else:
        os.mkdir(out_f)
    '''
    从对应出的测井数据中对测井数据进行进一步的筛选，对测井数据中的缺失字段进行填充
    :param test_f: 提取+对应后的测井数据所在文件夹
    :param out_f: 对测井数据进行处理后的结果存储路径
    :param out_name: 对测井数据进行处理后的结果的文件名称-所有测井文件合并到一个csv中
    :return:
    '''
    # 设置测井数据所在的文件夹路径
    path = test_f
    files = os.listdir(path)
    cejing_big = pd.DataFrame()
    count = 0
    # 遍历文件夹下各测井数据文件
    for f in files:
        path1 = os.path.join(path, f)
        print(path1)
        # 读取测井数据文件，去除异常值
        df1 = pd.read_csv(path1, encoding='utf_8_sig', engine='python', sep=',')
        df1 = df1.replace('-999.25', np.NAN)
        # 若该文件为空，就输出文件名继续遍历，否则就输出文件的shape
        if len(df1) == 0:
            print(f)
            continue
        print(df1.shape)
        count += 1
        # df1 = df1.sort_values(by='解释日期',ascending=False)
        # df1 = df1.drop_duplicates(["砂层顶深",'#DEPTH'],keep="first")
        # if(df1[df1['AC']>])
        df2 = pd.DataFrame()
        column_list = np.array(df1.columns).tolist()
        # print(column_list)
        # print(df1.shape)
        # 测井参数列填补，用GRC填充GR、CALI填充CAL等
        df2['DEPTH'] = df1['#DEPTH']
        if 'GR' in column_list:
            df2['GR'] = df1['GR']
        elif 'GRC' in column_list:
            df2['GR'] = df1['GRC']
        if 'CAL' in column_list:
            df2['CAL'] = df1['CAL']
        elif 'CALI' in column_list:
            df2['CAL'] = df1['CALI']
        if 'AC' in column_list:
            df2['AC'] = df1['AC']
        if 'DEN' in column_list:
            df2['DEN'] = df1['DEN']
        elif 'CDEN' in column_list:
            df2['DEN'] = df1['CDEN']
        if 'CNL' in column_list:
            df2['CNL'] = df1['CNL']
        elif 'CN' in column_list:
            df2['CNL'] = df1['CN']
        elif 'CN1' in column_list:
            df2['CNL'] = df1['CN1']

        # df2['电阻率'] = df1['电阻率']
        # if 'RS' in column_list:
        #     df2['RS'] = df1['RS']
        if 'RA04' in column_list:
            df2['RA04'] = df1['RA04']
        elif 'R04' in column_list:
            df2['RA04'] = df1['R04']
        elif 'R040' in column_list:
            df2['RA04'] = df1['R040']
        # if 'RXO' in column_list:
        #     df2['RXO'] = df1['RXO']
        # if 'RA4' in column_list:
        #     df2['RA4'] = df1['RA4']
        # elif 'R4' in column_list:
        #     df2['RA4'] = df1['R4']
        # 要观察是否各个测井文件中有文件头的一定是测了的，要不要加判断条件，ra45在文件头中且不为空
        if 'RA45' in column_list:
            df2['RA45'] = df1['RA45']
        elif 'R45' in column_list:
            df2['RA45'] = df1['R45']
        elif 'RA04' in column_list:
            df2['RA45'] = df1['RA04']
        elif 'RA045' in column_list:
            df2['RA45'] = df1['RA045']
        elif 'R045' in column_list:
            df2['RA45'] = df1['R045']
        # elif 'R04' in column_list:
        #     df2['RA45'] = df1['R04']
        # elif 'R040' in column_list:
        #     df2['RA45'] = df1['R040']
        if 'RXO' in column_list:
            df2['RXO'] = df1['RXO']
        elif 'RX0' in column_list:
            df2['RXO'] = df1['RX0']
        if 'RLL8' in column_list:
            df2['RLL8'] = df1['RLL8']
        elif 'LL8' in column_list:
            df2['RLL8'] = df1['LL8']
        if 'RIL8' in column_list:
            df2['RIL8'] = df1['RIL8']
        elif 'IL8' in column_list:
            df2['RIL8'] = df1['IL8']
        if 'RILD' in column_list:
            df2['RILD'] = df1['RILD']
        elif 'ILD' in column_list:
            df2['RILD'] = df1['ILD']
        if 'RLLD' in column_list:
            df2['RLLD'] = df1['RLLD']
        elif 'LLD' in column_list:
            df2['RLLD'] = df1['LLD']
        if 'RL3D' in column_list:
            df2['RL3D'] = df1['RL3D']
        elif 'L3D' in column_list:
            df2['RL3D'] = df1['L3D']
        if 'RL3S' in column_list:
            df2['RL3S'] = df1['RL3S']
        elif 'L3S' in column_list:
            df2['RL3S'] = df1['L3S']
        if 'RLLS' in column_list:
            df2['RLLS'] = df1['RLLS']
        elif 'LLS' in column_list:
            df2['RLLS'] = df1['LLS']
        if 'RILM' in column_list:
            df2['RILM'] = df1['RILM']
        if 'ILM' in column_list:
            df2['RILM'] = df1['ILM']
        if 'RD' in column_list:
            df2['Rd'] = df1['RD']
        elif 'Rd' in column_list:
            df2['Rd'] = df1['Rd']
        elif 'rd' in column_list:
            df2['Rd'] = df1['rd']
        if 'RT' in column_list:
            df2['RT'] = df1['RT']
        if 'GR' in column_list:
            df2['GR'] = df1['GR']
        if 'RA25' in column_list:
            df2['RA25'] = df1['RA25']
        elif 'R25' in column_list:
            df2['RA25'] = df1['R25']
        elif 'R250' in column_list:
            df2['RA25'] = df1['R250']
        # if 'MINV' in column_list:
        #     df2['MINV'] = df1['MINV']
        # if 'MNOR' in column_list:
        #     df2['MNOR'] = df1['MNOR']
        if 'SP' in column_list:
            df2['SP'] = df1['SP']
        elif 'SP1' in column_list:
            df2['SP'] = df1['SP1']
        # df2['同深度'] = df1['同深度']
        # df2['井号'] = f.split('@')[0]
        df2['cellid'] = df1['cellid']
        # df2['id'] = df1['id']
        # df2['泥浆电阻率'] = df1['泥浆电阻率']
        # df2['砂层顶深'] = df1['砂层顶深']
        # df2['砂层底深'] = df1['砂层底深']
        # df2['砂层厚度'] = df1['砂层厚度']
        # df2['解释日期'] = df1['解释日期']
        # df2['射孔情况'] = df1['射孔情况']
        # df2['射孔日期'] = df1['射孔日期']
        # df2['油层组'] = df1['油层组']
        # df2['解释结论标签'] = df1['解释结论标签']
        # df2['区块'] = df1['区块']
        # df2['潜力结果'] = df1['潜力结果']
        # df2['lu_index新'] = df1['lu_index新']
        # df2[['pro_index','射孔日期','射孔段顶深','射孔段底深','月产水量','月产气量','月产油量','生产层号','累积产气量','累积产水量','累积产油量','解释序号']] = df1[['pro_index',
        #                           '射孔日期','射孔段顶深','射孔段底深','月产水量','月产气量','月产油量','生产层号','累积产气量','累积产水量','累积产油量','解释序号']]

        # df2['含油级别旧'] = df1['含油级别旧']
        # df2['含油级别新'] = df1['含油级别新']
        # df2['岩石颜色'] = df1['岩石颜色']
        # df2['岩石名称'] = df1['岩石名称']
        # df2['颜色号'] = df1['颜色号']
        # df2['岩性描述'] = df1['岩性描述']
        # df2['荧光颜色'] = df1['荧光颜色']
        # df2['顶界井深新'] = df1['顶界井深新']
        # df2['底界井深新'] = df1['底界井深新']
        # df2['顶界井深旧'] = df1['顶界井深旧']
        # df2['底界井深旧'] = df1['底界井深旧']
        # df2['录井测井解释'] = df1['录井测井解释']
        # df2['渗透率'] = df1['渗透率']
        # df2['孔隙度'] = df1['孔隙度']
        # df2['泥质含量'] = df1['泥质含量']
        # 将各井填充好的测井数据拼接起来
        cejing_big = pd.concat([cejing_big, df2])
        # df2=df2.replace(0, np.nan, regex=True)
        # df2=df2.replace(-999.250, np.nan, regex=True)
    print('----------------------------------------------------------------------')
    print(cejing_big.columns)
    # cejing_big['砂层顶深'] = cejing_big['砂层顶深'].astype('str')
    # cejing_big['cellid'] = cejing_big['井号'].str.cat(cejing_big["砂层顶深"], sep='$')
    print('count不空的井', count)
    print('cejing_big的大小', cejing_big.shape)
    # cejing_big = cejing_big.dropna(subset=['SP','AC','MINV','MNOR'])
    # print('cejing_big小层数', len(list(cejing_big['cellid'].unique())))
    # 输出到文件路径
    out_p=os.path.join(out_f,out_name)
    cejing_big.to_csv(out_p, encoding='utf_8_sig', sep=',',
                      index=False)


def merge(test_data,layer_Data,out_p,out_name):
    '''

    :param test_data: 测井数据路径
    :param layer_Data: 生产数据路径
    :param out_p: 测井+小层合并后输出的路径
    :param out_name: 测井+小层合并后输出的文件名称
    :return:
    '''
    if os.path.exists(out_p):
        shutil.rmtree(out_p)
        os.mkdir(out_p)
    else:
        os.mkdir(out_p)

    # 将cejing_big与xiaoceng两个dataframe按照cellid这一列进行merge
    # cejing_big = pd.read_excel(r'E:\石油资料\港中分析方法迁移到港东\方法优化\训练集测井数据367测井数据.xlsx')
    files=os.listdir(test_data)
    for f in files:
        temp_test=os.path.join(test_data,f)
        print("temp_test",temp_test)
        cejing_big = pd.read_csv(temp_test)
        try:
            xiaoceng = pd.read_csv(layer_Data, encoding='utf_8_sig', sep=',')
        except:
            xiaoceng = pd.read_csv(layer_Data, encoding='gbk', sep=',')

        xiaoceng['cellid'] = xiaoceng['井号'].str.cat(xiaoceng['砂层顶深'].astype(str), sep='$')
        # xiaoceng =  xiaoceng[['AC','CAL','CNL','DEN','DEPTH','GR','RA04','RA25','RA45','RT','RXO','Rd','SP','cellid','解释结论标签','潜力具备潜力','井况','原因']]
        # print(len(list(cejing_big['井号'].unique())))

        df = pd.merge(xiaoceng, cejing_big, on='cellid', how='left')  #以cellid列为合并的规则，进行左连接：以左侧 DataFrame 的键为准，保留左侧 DataFrame 中的所有行，并将右侧 DataFrame 中匹配的行合并到一起。
        print(len(list(df['cellid'].unique())))
        print(df.columns)
        # print(len(list(df['井号'].unique())))
        # df = df.drop_duplicates(subset=['cellid'],keep='last')
        # print(df.shape)
        out_path=os.path.join(out_p,out_name)
        # df=df[['井号', '解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度', '泥质含量', '射孔情况', 'y_label', 'cellid','AC', 'CNL', 'DEN', 'DEPTH', 'GR','RA25','RT', 'SP']]
        df.to_csv(out_path, encoding='utf_8_sig', sep=',', index=False)
def celushengdata(data1, data2):
        # print(len(data1))
        # print(len(data2))
        # print(data2["JH"]+data2["顶深"])
        # data1:训练或者测试数据
        # data2:录井数据
        # print(data2.columns)
        data2["cellid"] = data2["JH"] + "$" + data2["顶深"].astype(str)
        data2 = data2[["cellid", "含油级别", "YGYS", '岩石描述']]
        # #饱含油:1
        # #饱含油:2
        # #饱含油:3
        # #饱含油:4
        # #饱含油:5
        # #饱含油:6

        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["饱含油"],value =1)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["富含油"],value =2)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["荧光"],value =3)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["油斑"],value =4)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["油迹"],value =5)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["油浸"],value =6)
        # # print(data2.head(10))
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["暗黄","暗黄色"],value =1)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["无色","无"],value =2)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["亮黄色"],value =3)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["棕黄","棕黄色"],value =4)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["黄白","黄白色"],value =5)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["淡黄","淡黄色"],value =6)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["黄","黄色"],value =7)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["浅黄","浅黄色"],value =8)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["黄褐","黄褐色","褐黄色"],value =9)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["亮黄"],value =10)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["蛋黄"],value =11)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["乳白","白"],value =12)
        # data2["YGYS"] = data2["YGYS"].replace(to_replace=["乳白", "白"], value=12)
        data3 = pd.merge(data1, data2, on='cellid')
        return data3
def select_param(data,file):   #测试集没有"y_label
    #（1）5RA25：AC、CNL、GR、SP、RA25
    # （2）6RA25：AC、CNL、DEN、GR、SP、RA25
    # （3）6RT：AC、CNL、DEN、GR、SP、RT
    #cal, den, ra04, ra45, rt, 区块，
    # if "训练" in file:
    #     data1=data[['井号','YGYS','岩石描述','含油级别','解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度', '泥质含量', '射孔情况', 'y_label', 'cellid','AC','CNL','DEPTH', 'GR','RA25','SP']]
    #     # data1[""]
    #     data2=data[['井号', 'YGYS','岩石描述','含油级别','解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度', '泥质含量', '射孔情况', 'y_label', 'cellid','AC','CNL','DEN', 'DEPTH', 'GR','RA25','SP']]
    #     data3=data[['井号', 'YGYS','岩石描述','含油级别','解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度', '泥质含量', '射孔情况', 'y_label', 'cellid','AC','CNL','DEN', 'DEPTH', 'GR','RT', 'SP']]
    # # data4=data[['井号', '解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度', '泥质含量', '射孔情况', 'y_label', 'cellid','AC','CAL','CNL','DEN', 'DEPTH', 'GR','RA25','RA04','RA45','RT', 'SP']]
    # else:
    #     data1 = data[ ['井号', 'YGYS','岩石描述','含油级别','解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度','泥质含量', '射孔情况', 'cellid', 'AC', 'CNL', 'DEPTH', 'GR', 'RA25', 'SP']]
    #     data2 = data[['井号','YGYS','岩石描述','含油级别', '解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度','泥质含量', '射孔情况', 'cellid', 'AC', 'CNL', 'DEN', 'DEPTH', 'GR', 'RA25', 'SP']]
    #     data3=data[['井号', 'YGYS','岩石描述','含油级别','解释序号', '解释日期', '油组', '层组', '小层号', '砂层号', '砂层顶深', '砂层底深', '砂层厚度', '电测解释结论', '孔隙度', '渗透率', '含油饱和度', '泥质含量', '射孔情况', 'cellid','AC','CNL','DEN', 'DEPTH', 'GR','RT', 'SP']]
    if "训练" in file:
        data1=data.dropna(axis=0, subset=['AC','CNL','DEPTH', 'GR','RA25','SP'])
        # print("data1", data1)
        data1=data1[["井号","DEPTH","cellid","y_label","GR","AC","CNL","CAL","DEN","RA25","RA04","RA45","SP","RT","砂层顶深","砂层底深","孔隙度","岩石名称_油气显示","含油级别","荧光颜色_油气显示","射孔情况","油组","层组"]]
        data2 = data.dropna(axis=0,subset=['AC', 'CNL', 'DEN', 'DEPTH', 'GR', 'RA25', 'SP'])
        data2 = data2[["井号", "DEPTH", "cellid", "y_label", "GR", "AC", "CNL", "CAL", "DEN", "RA25", "RA04", "RA45", "SP", "RT","砂层顶深", "砂层底深", "孔隙度", "岩石名称_油气显示", "含油级别", "荧光颜色_油气显示", "射孔情况", "油组", "层组"]]
        data3 = data.dropna(axis=0, subset=['AC', 'CNL', 'DEN', 'DEPTH', 'GR', 'RT', 'SP'])
        data3 = data3[["井号", "DEPTH", "cellid", "y_label", "GR", "AC", "CNL", "CAL", "DEN", "RA25", "RA04", "RA45", "SP", "RT","砂层顶深", "砂层底深", "孔隙度", "岩石名称_油气显示", "含油级别", "荧光颜色_油气显示", "射孔情况", "油组", "层组"]]
        data1["区块"]=""
        data2["区块"] = ""
        data3["区块"] = ""
    else:
        data1 = data.dropna(axis=0, subset=['AC', 'CNL', 'DEPTH', 'GR', 'RA25', 'SP'])
        # print("data1", data1)
        data1 = data1[
            ["井号", "DEPTH", "cellid", "GR", "AC", "CNL", "CAL", "DEN", "RA25", "RA04", "RA45", "SP", "RT",
             "砂层顶深", "砂层底深", "孔隙度", "岩石名称_油气显示", "含油级别", "荧光颜色_油气显示", "射孔情况", "油组", "层组"]]
        data2 = data.dropna(axis=0, subset=['AC', 'CNL', 'DEN', 'DEPTH', 'GR', 'RA25', 'SP'])
        data2 = data2[
            ["井号", "DEPTH", "cellid","GR", "AC", "CNL", "CAL", "DEN", "RA25", "RA04", "RA45", "SP", "RT",
             "砂层顶深", "砂层底深", "孔隙度", "岩石名称_油气显示", "含油级别", "荧光颜色_油气显示", "射孔情况", "油组", "层组"]]
        data3 = data.dropna(axis=0, subset=['AC', 'CNL', 'DEN', 'DEPTH', 'GR', 'RT', 'SP'])
        data3 = data3[["井号", "DEPTH", "cellid", "GR", "AC", "CNL", "CAL", "DEN", "RA25", "RA04", "RA45", "SP", "RT","砂层顶深", "砂层底深", "孔隙度", "岩石名称_油气显示", "含油级别", "荧光颜色_油气显示", "射孔情况", "油组", "层组"]]
        data1["区块"] = ""
        data2["区块"] = ""
        data3["区块"] = ""
    # print(data1.head())
    return data1,data2,data3

def addwater(data):
    youList = data[data['y_label'] == '油层']['cellid'].drop_duplicates().values.tolist()
    # print("油数量: ", len(youList))
    shuiList = data[data['y_label'] == '水层']['cellid'].drop_duplicates().values.tolist()
    # print("水数量: ", len(shuiList))
    # 遍历水层造水
    ans = pd.DataFrame()
    col = ['AC', 'GR', 'SP', 'CNL']
    cnt = 0
    for j in range(4):
        # 对哪一列进行改造
        colMul = col[j]
        # 遍历水层
        for k in range(len(shuiList)):
            # 拿到一个水层的id
            id = shuiList[k]
            shui = data[data['cellid'] == id]
            # 改造
            lst = shui[colMul].values.tolist()
            List = [1.1, 0.9, 1.05]
            # c=choice(List)
            for c in List:
                newCol = []
                for ls in lst:
                    newCol.append(ls * c)
                shui[colMul] = newCol
                # 修改cellid，因为是新造的水层
                shui["cellid"] = id + "-" + colMul
                # 造完放起来
                # 拼接起来
                ans = pd.concat([ans, shui])
                cnt += 1
                # if cnt >= (len(youList) - len(shuiList))*2/3:
                if cnt+len(shuiList)>=len(youList)*2/3:
                    break
            if cnt + len(shuiList) >= len(youList) * 2 / 3:
                break
        if cnt+len(shuiList)>=len(youList)*2/3:
            break
    data = pd.concat([data, ans])
    print(cnt)
    data = data.replace("油层", "1")
    data = data.replace("水层", "0")
    return data
def luxiaocengmerge(xiaocengdata,ludata):  #对比测井数据中的顶深低深，添加了含油级别，岩石名称_油气显示，荧光颜色_油气显示三列数据
    xiaocengdata["含油级别"] = ""
    xiaocengdata["岩石名称_油气显示"] = ""
    xiaocengdata["荧光颜色_油气显示"] = ""
    # print(xiaocengdata[["砂层顶深","砂层底深"]].head())
    print(xiaocengdata.columns)
    xiaocengdata["砂层顶深"]=xiaocengdata["砂层顶深"].astype("float")
    xiaocengdata["砂层底深"] = xiaocengdata["砂层底深"].astype("float")
    topdepth=""
    bottomdepth=""
    rock=""
    color=""
    if "顶界井深_油气显示" in list(ludata.columns):
        topdepth="顶界井深_油气显示"
        ludata["顶界井深_油气显示"] = ludata["顶界井深_油气显示"].astype("float")
    if "顶深" in list(ludata.columns):
        topdepth="顶深"
        ludata["顶深"] = ludata["顶深"].astype("float")
    if "底界井深_油气显示"in list(ludata.columns):
        bottomdepth="底界井深_油气显示"
        ludata["底界井深_油气显示"] = ludata["底界井深_油气显示"].astype("float")
    if "底深" in list(ludata.columns):
        bottomdepth = "底深"
        ludata["底深"] = ludata["底深"].astype("float")
    if "岩石名称_油气显示" in list(ludata.columns):
        rock="岩石名称_油气显示"
    if "岩石描述"in list(ludata.columns):
        rock = "岩石描述"
    if "荧光颜色_油气显示" in list(ludata.columns):
        color="荧光颜色_油气显示"
    if "YGYS"in list(ludata.columns):
        color = "YGYS"
    for i in range(len(xiaocengdata)):
        for j in range(len(ludata)):
            # print("i",i,"j",j)
            # print(xiaocengdata.loc[1,"砂层顶深"],xiaocengdata.loc[1,"砂层底深"])
            if (xiaocengdata.loc[i,"砂层顶深"]>=ludata.loc[j,topdepth])and(xiaocengdata.loc[i,"砂层底深"]<=ludata.loc[j,bottomdepth]):
                xiaocengdata.loc[i,"含油级别"]=ludata.loc[j,"含油级别"]
                xiaocengdata.loc[i, "岩石名称_油气显示"] = ludata.loc[j, rock]
                xiaocengdata.loc[i, "荧光颜色_油气显示"] = ludata.loc[j, color]
                break
            elif (xiaocengdata.loc[i,"砂层顶深"]>=ludata.loc[j,topdepth])and(xiaocengdata.loc[i,"砂层顶深"]<=ludata.loc[j,bottomdepth]):
                xiaocengdata.loc[i, "含油级别"] = ludata.loc[j, "含油级别"]
                xiaocengdata.loc[i, "岩石名称_油气显示"] = ludata.loc[j, rock]
                xiaocengdata.loc[i, "荧光颜色_油气显示"] = ludata.loc[j, color]
                break
            elif (xiaocengdata.loc[i,"砂层顶深"]<=ludata.loc[j,topdepth])and(xiaocengdata.loc[i,"砂层底深"]>=ludata.loc[j,topdepth]):
                xiaocengdata.loc[i, "含油级别"] = ludata.loc[j, "含油级别"]
                xiaocengdata.loc[i, "岩石名称_油气显示"] = ludata.loc[j, rock]
                xiaocengdata.loc[i, "荧光颜色_油气显示"] = ludata.loc[j, color]
                break
            elif (abs(xiaocengdata.loc[i,"砂层顶深"]-ludata.loc[j,topdepth])<=0.5)or(abs(xiaocengdata.loc[i,"砂层底深"]-ludata.loc[j,bottomdepth])<=0.5):
                xiaocengdata.loc[i, "含油级别"] = ludata.loc[j, "含油级别"]
                xiaocengdata.loc[i, "岩石名称_油气显示"] = ludata.loc[j, rock]
                xiaocengdata.loc[i, "荧光颜色_油气显示"] = ludata.loc[j, color]
                break

    return xiaocengdata
def main2(path1,path2,path3,path4):
####
    '''
    从小层数据中选取训练数据和测试集
    并且为训练数据打上水层油层标签
    '''
    readxcpath =path1
    readshpath=path2
    fp=path3
    lujingfilename = path4
    savexcpath = path+"\\out_put_射孔小层数据\\已射孔小层.csv"#马东小层射孔数据
    savexcpathp = path+"\\未射孔小层数据\\未射孔小层.csv"#马东小层未射孔数据（即待预测小层）
    savexcpath1 = path+"\\out_put_射孔油层数据\\已射孔油层.csv"#马东小层打油水层标签的数据
    xcdata = pd.read_csv(readxcpath, encoding="utf_8")
    # xcdata=pd.read_excel(readxcpath)
    scdata = pd.read_csv(readshpath, encoding="utf_8")
    # scdata=pd.read_excel(readshpath)
    data1,data1p=skoilceng(xcdata)#返回射孔数据和非射孔数据
    data1.to_csv(savexcpath,encoding="utf_8",index=False)
    data1p.to_csv(savexcpathp, encoding="utf_8", index=False)
    data2=dizuoilceng(data1,scdata)#将训练集数据打上油水层标签
    data2.to_csv(savexcpath1,encoding="utf-8",index=False)
# ####
#     '''
#     选取同分布测试数据
#     '''
    DATA_PATH = path+'\\out_put_射孔油层数据\\'  # 已核实小层文件目录
    DATA_PATH1 = path+'\\未射孔小层数据\\'  # 带预测小层文件目录
    presmalayfilename = "未射孔小层.csv"
    # presmalaysheet_name = "岩石筛选44290"
    verifyfilename = "已射孔油层.csv"
    EDN_PATH = path+'\\待预测小层数据\\'  # 存储的和训练集同分布的带预测小层路径
    endfilename = "待预测小层数据.csv"
    EDN_PATH1 = path+'\\小层训练数据\\'  # 存储的和训练集同分布的带预测小层路径
    endfilename1 = "小层训练数据.csv"
    # df_train = pd.read_excel(DATA_PATH + presmalayfilename,sheet_name=presmalaysheet_name)#带预测小层文件作为训练集
    # df_test = pd.read_excel(DATA_PATH + verifyfilename)#将训练集数据作为测试集
    df_train = pd.read_csv(DATA_PATH1 + presmalayfilename, encoding="utf_8")
    df_train = df_train.drop_duplicates(subset=['cellid'])
    df_test = pd.read_csv(DATA_PATH + verifyfilename, encoding="utf_8")
    df_test = df_test.drop_duplicates(subset=['cellid'])
    # random_df = apprix_df.sample(frac=0.4, random_state=60)
    print("df_train长度", len(df_train))
    print("df_test长度", len(df_test))
#     '''
#     数据处理
#     '''
    X_number, df_adv, df_trainp, df_testp = dataprocess(df_train, df_test)
    # print("df_adv长度", len(df_adv))
#     '''
#     模型构建及预测
#     '''
    preds_adv = modelbuilding(X_number, df_adv)
    print("preds_adv长度", len(preds_adv))
#     '''
#     选取预测概率最高的待小层数据
#     '''
    df_train_2 = selectsamlay(df_testp, preds_adv)
    df_trainp.to_csv(EDN_PATH1 + endfilename1, encoding='utf_8', index=False)
    df_train_2.to_csv(EDN_PATH + endfilename, encoding='utf_8', index=False)
####
    '''
    将训练集与测试集小层数据与录井数据合并
    '''
    DATA_PATH = path+'\\小层训练数据\\'  # 已核实小层文件目录
    DATA_PATH1 = path+'\\待预测小层数据\\'   # 带预测小层文件目录
    filename = "小层训练数据.csv"  #
    # presmalaysheet_name = "岩石筛选44290"
    filename1 = "待预测小层数据.csv"  # 训练数据
    endfilename1=path+"\\小层与录井数据融合\\小层与录井融合训练数据.csv"
    endfilename2=path+"\\小层与录井数据融合\\小层与录井融合测试数据.csv"
    # # '''
    # # 文件读取
    # # '''
    # # df_train = pd.read_excel(DATA_PATH + presmalayfilename,sheet_name=presmalaysheet_name)#带预测小层文件作为训练集
    # # df_test = pd.read_excel(DATA_PATH + verifyfilename)#将训练集数据作为测试集
    df_train = pd.read_csv(DATA_PATH + filename, encoding="utf_8")
    df_train = df_train.drop_duplicates(subset=['cellid'])
    df_train=df_train.reset_index(drop=True)
    df_test = pd.read_csv(DATA_PATH1 + filename1, encoding="utf_8")
    df_test = df_test.drop_duplicates(subset=['cellid'])
    df_test=df_test.reset_index(drop=True)
    ludata=pd.read_csv(lujingfilename, encoding="utf-8")
    # ludata=pd.read_excel(lujingfilename)
    train=luxiaocengmerge(df_train,ludata)
    train.to_csv(endfilename1,encoding="utf_8",index=False)
    test=luxiaocengmerge(df_test,ludata)
    test.to_csv(endfilename2,encoding="utf_8",index=False)
#
# ####
#     '''
#     提取测井数据并将训练集与测试集小层+录井与测井合并
#     '''
    layer_p=path+'\\小层与录井数据融合\\小层与录井融合训练数据.csv'
    layer_p1=path+'\\小层与录井数据融合\\小层与录井融合测试数据.csv'
    out_p1=path+'\\out_put_1\\' #txt转csv后的存储路径
    txt2pd(fp,out_p1)
    # '''
    # 提取测井数据
    # '''
    # txt2merge_csv(fp,out_p1)
    print('已成功：提取测井数据\n')
    # '''
    # 将测井数据与小层数据进行对应
    # '''
    out_p2=path+'\\out_put_2\\'      #提取小层数据对应的测井数据后，结果的存储路径
    out_p2p =path+'\\out_put_2p\\'  # 提取小层数据对应的测井数据后，结果的存储路径
    select_test_data(out_p1,layer_p,out_p2)
    select_test_data(out_p1, layer_p1, out_p2p)
    print('已成功：将测井数据与小层数据进行对应\n')
    # '''
    # 对测井数据进行填充和合并
    # '''
    out_p3=path+'\\out_put_3\\'
    out_p3p = path+'\\out_put_3p\\'
    out_name1='整理后的测井数据.csv'
    concat_test(out_p2,out_p3,out_name1)
    concat_test(out_p2p, out_p3p, out_name1)
    print('已成功：对测井数据进行填充和合并\n')
    # # '''
    # # 将测井数据与小层数据进行合并
    # # # '''
    out_p4 = path+'\\out_put_4\\'  # 提取小层数据对应的测井数据后，结果的存储路径
    out_p4p = path+'\\out_put_5\\'  # 待预测小层数据对应的测井数据后，结果的存储路径
    out_name2='测录小层融合训练数据.csv'
    out_name3 = '测录小层融合测试数据.csv'
    merge(out_p3,layer_p,out_p4,out_name2)
    print('已成功：将测井数据与小层训练数据进行合并\n')
    merge(out_p3p, layer_p1, out_p4p, out_name3)
    print('已成功：将测井数据与小层数据进行合并\n')
####
    # '''
    # 训练数据和测试数据小层+测井与录井结合
    # '''
    # file1=path+"\\out_put_4\\小层+测井数据_train_20220808.csv"#训练数据
    # file2=path+"\\out_put_4p\\小层+测井数据_test_20220808.csv"#测试数据
    # endfile1=path+"\\测录小层融合数据\\测录小层融合训练数据.csv"
    # endfile2=path+"\\测录小层融合数据\\测录小层融合测试数据.csv"
    # data1 = pd.read_csv(lujingfilename, encoding="gbk")
    # data2 = pd.read_csv(file1, encoding="utf_8")
    # data3 = pd.read_csv(file2, encoding="utf_8")
    # data4=celushengdata(data2,data1)
    # data4.to_csv(endfile1, encoding="utf_8", index=False)
    # data5=celushengdata(data3, data1)
    # data5.to_csv(endfile2, encoding="utf_8", index=False)
# ####
#     '''
#     选参数
#     '''
    filenames=[path+"\\out_put_4\\测录小层融合训练数据",path+"\\out_put_5\\测录小层融合测试数据"]
    for file in filenames:
        data=pd.read_csv(file+".csv",encoding="utf_8")
        data1,data2,data3=select_param(data,file)
        # print("11111",data1.head())
        if "训练" in file:
            data1.to_csv(path+"\\测录小层融合数据\\小层+测井数据+录井+训练+5RA25.csv",encoding="utf_8",index=False)
            data2.to_csv(path+"\\测录小层融合数据\\小层+测井数据+录井+训练+6RA25.csv",encoding="utf_8",index=False)
            data3.to_csv(path + "\\测录小层融合数据\\小层+测井数据+录井+训练+6RT.csv", encoding="utf_8", index=False)
        else:
            # print("2222",data1.head())
            data1=data1.drop(columns=["井号"])
            data2 = data2.drop(columns=["井号"])
            data3 = data3.drop(columns=["井号"])
            data1.rename(
                columns={"cellid": "cellid", "DEPTH": "depth","GR": "gr", "AC": "ac", "CNL": "cnl",
                         "CAL": "cal", "DEN": "den", "RA25": "ra25", "RA04": "ra04", "RA45": "ra45", "SP": "sp",
                         "RT": "rt", "砂层顶深": "top_depth", "砂层底深": "down_depth", "孔隙度": "por", "岩石名称_油气显示": "rock",
                         "含油级别": "hyjb", "荧光颜色_油气显示": "ygys", "射孔情况": "shoted", "区块": "block", "油组": "group",
                         "层组": "layer_group"}, inplace=True)
            # print(data1)
            data1["ra4"]=""
            data2.rename(
                columns={"cellid": "cellid", "DEPTH": "depth", "GR": "gr", "AC": "ac", "CNL": "cnl",
                         "CAL": "cal", "DEN": "den", "RA25": "ra25", "RA04": "ra04", "RA45": "ra45", "SP": "sp",
                         "RT": "rt", "砂层顶深": "top_depth", "砂层底深": "down_depth", "孔隙度": "por", "岩石名称_油气显示": "rock",
                         "含油级别": "hyjb", "荧光颜色_油气显示": "ygys", "射孔情况": "shoted", "区块": "block", "油组": "group",
                         "层组": "layer_group"}, inplace=True)
            data2["ra4"] = ""
            data3.rename(
                columns={"cellid": "cellid", "DEPTH": "depth", "GR": "gr", "AC": "ac", "CNL": "cnl",
                         "CAL": "cal", "DEN": "den", "RA25": "ra25", "RA04": "ra04", "RA45": "ra45", "SP": "sp",
                         "RT": "rt", "砂层顶深": "top_depth", "砂层底深": "down_depth", "孔隙度": "por", "岩石名称_油气显示": "rock",
                         "含油级别": "hyjb", "荧光颜色_油气显示": "ygys", "射孔情况": "shoted", "区块": "block", "油组": "group",
                         "层组": "layer_group"}, inplace=True)
            data3["ra4"] = ""
            columns1 = ["cellid", "depth", "ac", "den", "cnl", "cal", "gr", "sp", "ra04", "ra4", "ra25",
                        "ra45", "rt", "top_depth", "down_depth", "por", "rock", "hyjb", "ygys", "shoted", "block",
                        "group", "layer_group"]
            data1 = data1[(data1["rt"] < 5) & (data1["rt"] > 0)]
            data2 = data2[(data1["rt"] < 5) & (data2["rt"] > 0)]
            data3 = data3[(data1["rt"] < 5) & (data3["rt"] > 0)]
            data1.to_csv(path+"\\测录小层融合数据\\小层+测井数据+录井+测试+5RA25.csv", encoding="utf_8", index=False,columns=columns1)
            data2.to_csv(path+"\\测录小层融合数据\\小层+测井数据+录井+测试+6RA25.csv", encoding="utf_8", index=False,columns=columns1)
            data3.to_csv(path+"\\测录小层融合数据\\小层+测井数据+录井+测试+6RT.csv", encoding="utf_8", index=False,columns=columns1)
            data1.to_csv(path+"\\数据预处理后的训练集和测试集\\小层+测井数据+录井+测试+5RA25.csv", encoding="utf_8", index=False,columns=columns1)
            data2.to_csv(path+"\\数据预处理后的训练集和测试集\\小层+测井数据+录井+测试+6RA25.csv", encoding="utf_8", index=False,columns=columns1)
            data3.to_csv(path+"\\数据预处理后的训练集和测试集\\小层+测井数据+录井+测试+6RT.csv", encoding="utf_8", index=False,columns=columns1)
# ####
#     '''
#     确定油水比添加水层
#     '''
    readfilename=path+"\\测录小层融合数据\\小层+测井数据+录井+训练+"
    savefilename=path+"\\增加水层数据\\小层+测井数据+录井+训练+"
    savefilename1=path+"\\数据预处理后的训练集和测试集\\小层+测井数据+录井+训练+"
    name = ['5RA25','6RA25','6RT']
    for i in name:
        f = open(readfilename+i+".csv", encoding='utf-8')
        data = pd.read_csv(f)
        # print(len(data))
        data1=addwater(data)
        # print(data1["cellid"])
        data1 = data1.drop(columns=["井号"])
        data1.rename(columns={"cellid":"cellid","DEPTH":"depth","y_label":"y_lable","GR":"gr","AC":"ac","CNL":"cnl","CAL":"cal","DEN":"den","RA25":"ra25","RA04":"ra04","RA45":"ra45","SP":"sp","RT":"rt","砂层顶深":"top_depth","砂层底深":"down_depth","孔隙度":"por","岩石名称_油气显示":"rock","含油级别":"hyjb","荧光颜色_油气显示":"ygys","射孔情况":"shoted","区块":"block","油组":"group","层组":"layer_group"}, inplace=True)
        data1["ra4"] = ""
        columns1=["cellid","depth","y_lable","ac","den","cnl","cal","gr","sp","ra04","ra4","ra25","ra45","rt","top_depth","down_depth","por","rock","hyjb","ygys","shoted","block","group","layer_group"]
        data1 = data1[(data1["rt"] < 5) & (data1["rt"] > 0)]
        data1.to_csv(savefilename+i + "-addShui.csv", encoding='utf_8_sig', index=False,columns=columns1)
        data1.to_csv(savefilename1 + i + "-addShui.csv", encoding='utf_8_sig', index=False,columns=columns1)
path=os.getcwd()
path1=path+"\\data\\小层数据\\"
path2=path+"\\data\\生产数据\\"
# path3=path+"data\\测井数据"
path3=path+"\\data\\录井数据\\"
filename1=os.listdir(path1)[0]
filename2=os.listdir(path2)[0]
filename3=os.listdir(path3)[0]
print("数据路径",filename1)
readxcpath=path1+filename1
# print(filename1)
readshpath=path2+filename2
lujingfilename=path3+filename3
# readxcpath = "./data\\小层数据\\港中小层数据总表.csv"#马东小层数据
# readshpath = "./data\\生产数据\\港中生产数据总表.csv"#马东油井生产数据
fp=path+'\\data\\测井数据\\'#测井数据txt存放文件夹
# lujingfilename = "./data\\录井数据\\油气显示统计表.csv"#录井数据
main2(readxcpath,readshpath,fp,lujingfilename)
print("运行已完成")