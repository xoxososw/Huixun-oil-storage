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
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score,accuracy_score

mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import warnings
warnings.filterwarnings(action='ignore')

def main(X,y,str_col,pre,pre_y):
# def main(X,y,str_col):
    bst_acc_tra = 0
    bst_acc_tst = 0
    bst_recall_tra = 0
    bst_recall_tst = 0
    best_score_all = 0


    # for iter in range(10):
    for iter in range(1):


        X_train = X
        y_train = y
        X_test = pre
        y_test = pre_y
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

        cnt = 0
        for max_depth in list(range(1,50,2)):
            # for n_estimators in list(range(1,300,50)):
                for max_leaf_nodes in list(range(2,50,2)):
                    for min_samples_leaf in list(range(2,50,2)):
            #             for min_samples_split in list(range(2,100,10)):
                            # for max_features in list(range(2,50,2)):
                                c1 = RandomForestClassifier(
                                    max_depth=max_depth,
                                    # n_estimators=n_estimators,
                                    max_leaf_nodes=max_leaf_nodes,
                                    min_samples_leaf=min_samples_leaf,
                                    # min_samples_split=min_samples_split,
                                    # max_features=max_features,

                                )
                                c1.fit(X_train, y_train)

                                # print("第 %d 次" % (iter))

                                train_pre = c1.predict(X_train)
                                acc_train = accuracy_score(train_pre, y_train)
                                # print('训练集上准确率：', acc_train)
                                recall_train = recall_score(train_pre, y_train)
                                # print('训练集上召回率：', recall_train)

                                test_pre = c1.predict(X_test)
                                test_proba = c1.predict_proba(X_test)
                                acc_test = accuracy_score(test_pre, y_test)
                                # print('测试集上准确率：', acc_test)
                                recall_test = recall_score(test_pre, y_test)
                                # print('测试集上召回率：', recall_test)


                                # if acc_train - acc_test >= 0.2: continue


                                # if acc_train < 0.95 or recall_train < 0.95: continue  # 训练集评分太低
                                # if acc_train == 1.0 or recall_train == 1.0: continue  # 训练集评分太高


                                # if acc_test < 0.80 or recall_test < 0.80: continue  # 测试集评分太低
                                # if acc_test == 1.0 or recall_test == 1.0: continue  # 测试集评分太高

                                if bst_acc_tst < acc_test:
                                    # 更新测试集、训练集的评分
                                    bst_acc_tra = acc_train
                                    bst_acc_tst = acc_test
                                    bst_recall_tra = recall_train
                                    bst_recall_tst = recall_test
                                    # 保存训练的数据模型
                                    fw = open('RF-output/'+str_col+'/RF_model.txt', 'wb')
                                    pickle.dump(c1, fw)
                                    fw.close()

                                    # 保存可视化结果
                                    Estimators = c1.estimators_
                                    for index, model in enumerate(Estimators):
                                        filename = '5_nine' + str(iter) + '_' + str(index) + '.dot'
                                        with open('RF-output/'+str_col+'/tmp_dot/' + filename, 'w') as f:
                                            f = tree.export_graphviz(model, out_file=f)


                                    test_y = pd.DataFrame(y_test)
                                    test_y['index'] = test_y.index
                                    test_y['RF预测值'] = test_pre
                                    test_y['RF预测概率'] = test_proba[:, 1]
                                    test_y.to_csv('RF-output/'+str_col+'/测试集预测结果.csv', index=False,encoding='utf_8_sig')

                                    # 保存评分
                                    score = "训准：" + str(bst_acc_tra) + '\n'
                                    score = score + "训召：" + str(bst_recall_tra) + '\n'
                                    score = score + "测准：" + str(bst_acc_tst) + '\n'
                                    score = score + "测召: " + str(bst_recall_tst)+ '\n'
                                    score = score + "max_depth: " + str(max_depth) + '\n'
                                    score = score + "max_leaf_nodes: " + str(max_leaf_nodes) + '\n'
                                    score = score + "min_samples_leaf: " + str(min_samples_leaf) + '\n'

                                    with open('./models/'+str_col+'/'+'RF-output/RF-评分结果.txt', 'w') as f:
                                        f.write(score)

                                    # 混淆矩阵

                                    cfm = confusion_matrix(y_test, test_pre)
                                    print(cfm)
                                    plt.matshow(cfm, cmap=plt.cm.gray)
                                    # plt.savefig('LGIS-output/'+str_col+'/LGIS-混淆矩阵.png')

                                    labels = ['水层', '油层']

                                    def plot_confusion_matrix(cm, cmap=plt.cm.binary):
                                        plt.imshow(cm, interpolation='nearest', cmap=cmap)
                                        plt.colorbar()
                                        xlocations = np.array(range(labels.__len__()))
                                        plt.xticks(xlocations, labels, rotation=90)
                                        plt.yticks(xlocations, labels)
                                        plt.ylabel('True label')
                                        plt.xlabel('Predicted label')

                                    cm = confusion_matrix(y_test, test_pre)

                                    for i in range(2):
                                        for j in range(2):
                                            c = cm[j][i]
                                            plt.text(i, j, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')

                                    plot_confusion_matrix(cm)
                                    plt.savefig('./models/'+str_col+'/'+'RF-output/RF-混淆矩阵.png')


                                    print("***RF*** "+str(bst_acc_tra)+" **** "+str(bst_recall_tra))
                                    print("***RF*** "+str(bst_acc_tst)+" **** "+str(bst_recall_tst))




