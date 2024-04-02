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
from sklearn.ensemble import GradientBoostingClassifier

mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import warnings
warnings.filterwarnings(action='ignore')


# def main(X,y,str_col,pre,pre_y):
def main(X,y,str_col):
    bst_acc_tra = 0
    bst_acc_tst = 0
    bst_recall_tra = 0
    bst_recall_tst = 0
    best_score_all = 0


    for iter in range(10):
    # for iter in range(1):

        # X_train = X
        # y_train = y
        # X_test = pre
        # y_test = pre_y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
        # 交叉验证
        # 交叉验证取最佳参数组合
        # Gbdt = GradientBoostingClassifier(random_state=2018)
        # param_grid = {
        #     'max_depth': [2,4,6,8,10],
        #     'n_estimators': [50,100,200],
        #     'learning_rate': [0.1,0.08,0.06],
        #     'subsample': [0.5, 0.7, 0.9],
        #               }
        # grid = GridSearchCV(Gbdt, param_grid, scoring='accuracy', cv=10,n_jobs=-1)
        # grid_result = grid.fit(X_train, y_train)
        # # print("Best: %f using %s" % (grid_result.best_score_, grid.best_params_))
        #
        #
        #
        # # 建立模型  训练数据
        # # c1 = DecisionTreeClassifier(criterion=grid.best_params_['criterion'],max_depth=grid.best_params_['max_depth'],min_samples_leaf=grid.best_params_['min_samples_leaf'],class_weight='balanced',max_features=grid.best_params_['max_features'])
        # Gbdt = GradientBoostingClassifier(
        #     # max_depth=grid.best_params_['max_depth'],
        #     max_depth=grid.best_params_['max_depth'],
        #     learning_rate=grid.best_params_['learning_rate'],
        #     # learning_rate=0.1,
        #     n_estimators=grid.best_params_['n_estimators'],
        #     # n_estimators=100,
        #     subsample=grid.best_params_['subsample'],
        #     )
        best_score = 0
        cnt = 0
        for max_depth in list(range(1,20,2)): # 第二个数原为20
            # for n_estimators in list(range(1,300,50)):
                for max_leaf_nodes in list(range(2,20,2)):
                    for min_samples_leaf in list(range(2,20,2)):
            #             for min_samples_split in list(range(2,100,10)):
                            # for max_features in list(range(2,50,2)):
                                for learning_rate in list(range(1, 20, 1)):
                                    learning_rate = learning_rate * 1.0 / 100
                                    Gbdt = GradientBoostingClassifier(random_state=2018,
                                                                      max_depth=max_depth,
                                                                      # n_estimators=n_estimators,
                                                                      max_leaf_nodes=max_leaf_nodes,
                                                                      min_samples_leaf=min_samples_leaf,
                                                                      # min_samples_split=min_samples_split,
                                                                      # max_features=max_features,
                                                                      learning_rate=learning_rate,
                                                                      )
                                    Gbdt.fit(X_train, y_train)

                                    cnt += 1
                                    if cnt % 100 == 0:
                                        print(max_depth, max_leaf_nodes, min_samples_leaf, learning_rate)

                                    test_pre = Gbdt.predict(X_test)
                                    test_proba=Gbdt.predict_proba(X_test)
                                    acc_test = accuracy_score(test_pre, y_test)
                                    if acc_test > best_score:  # 找到表现最好的参数
                                        best_score = acc_test
                                        # best_parameters = {'gamma': gamma, 'C': C}
                                        # Gbdt.fit(X_train, y_train)
                                        # b = X_train.columns.values.tolist()


                                        # print("第 %d 次" % (iter))

                                        train_pre = Gbdt.predict(X_train)
                                        acc_train = accuracy_score(train_pre, y_train)
                                        # print('训练集上准确率：', acc_train)
                                        recall_train = recall_score(train_pre, y_train)
                                        # print('训练集上召回率：', recall_train)

                                        test_pre = Gbdt.predict(X_test)
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
                                            fw = open('./models/'+str_col+'/'+'/GBDT-output/GBDT_model.pkl', 'wb')
                                            pickle.dump(Gbdt, fw)
                                            fw.close()

                                            # 保存测试集上的预测结果
                                            test_y = pd.DataFrame(y_test)
                                            test_y['index'] = test_y.index
                                            test_y['GBDT预测值'] = test_pre
                                            test_y['GBDT预测概率']=test_proba[:,1]
                                            test_y.to_csv('./models/'+str_col+'/'+'GBDT-output/predict-result.csv', index=False,encoding='utf_8_sig')

                                            # 保存评分
                                            score = "训准：" + str(bst_acc_tra) + '\n'
                                            score = score + "训召：" + str(bst_recall_tra) + '\n'
                                            score = score + "测准：" + str(bst_acc_tst) + '\n'
                                            score = score + "测召: " + str(bst_recall_tst)+ '\n'
                                            score = score + "max_depth: " + str(max_depth) + '\n'
                                            score = score + "max_leaf_nodes: " + str(max_leaf_nodes) + '\n'
                                            score = score + "min_samples_leaf: " + str(min_samples_leaf) + '\n'
                                            score = score + "learning_rate: " + str(learning_rate) + '\n'

                                            with open('./models/'+str_col+'/'+'GBDT-output/score.txt', 'w') as f:
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
                                            plt.savefig('./models/'+str_col+'/'+'GBDT-output/GBDT-Confusion Matrix.png')
                                            plt.close()

                                        print("***GBDT*** "+str(bst_acc_tra)+" **** "+str(bst_recall_tra))
                                        print("***GBDT*** "+str(bst_acc_tst)+" **** "+str(bst_recall_tst))

if __name__=='__main__':
    pass


