# -*- coding: gbk -*-
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

from xgboost.sklearn import XGBClassifier

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
        # ������֤
        # ������֤ȡ��Ѳ������
        # Xgbc = XGBClassifier(random_state=2018, n_jobs=-1, n_estimator=30,
        #                      )
        # param_grid = {
        #     # 'n_estimators': [50,100,200],
        #     'learning_rate': [0.1,0.2,0.3],
        #     'max_depth': [2,4,6,8,10],
        #     'subsample': [0.5,0.7,0.9],
        #     # 'colsample_bytree':[0.6,0.8,1],
        #               }
        # grid = GridSearchCV(Xgbc, param_grid, scoring='accuracy', cv=10,n_jobs=-1)
        # grid_result = grid.fit(X_train, y_train)
        # # print("Best: %f using %s" % (grid_result.best_score_, grid.best_params_))

        # ����ģ��  ѵ������
        cnt = 0
        for max_depth in list(range(1,50,2)):
            # for n_estimators in list(range(1,300,50)):
            # for booster in ['gbtree', 'gblinear', 'dart']:
            #         for min_samples_leaf in list(range(2,50,2)):
                #         for min_samples_split in list(range(2,100,10)):
                            # for max_features in list(range(2,50,2)):
                                for learning_rate in list(range(1, 50, 1)):
                                    learning_rate = learning_rate * 1.0 / 100
                                    Xgbc = XGBClassifier(
                                        max_depth=max_depth,
                                        # n_estimators=n_estimators,
                                        # max_leaf_nodes=max_leaf_nodes,
                                        # min_samples_leaf=min_samples_leaf,
                                        # min_samples_split=min_samples_split,
                                        # max_features=max_features,
                                        # booster=booster,
                                        learning_rate=learning_rate
                                        )
                                    Xgbc.fit(X_train, y_train)
                                    cnt += 1

                                    # print("�� %d ��" % (iter))

                                    train_pre = Xgbc.predict(X_train)
                                    acc_train = accuracy_score(train_pre, y_train)
                                    # print('ѵ������׼ȷ�ʣ�', acc_train)
                                    recall_train = recall_score(train_pre, y_train)
                                    # print('ѵ�������ٻ��ʣ�', recall_train)

                                    test_pre = Xgbc.predict(X_test)
                                    test_proba = Xgbc.predict_proba(X_test)
                                    acc_test = accuracy_score(test_pre, y_test)
                                    # print('���Լ���׼ȷ�ʣ�', acc_test)
                                    recall_test = recall_score(test_pre, y_test)
                                    # print('���Լ����ٻ��ʣ�', recall_test)

                                    # if acc_train - acc_test >= 0.2: continue

                                    # if acc_train < 0.95 or recall_train < 0.95: continue  # ѵ��������̫��
                                    # if acc_train == 1.0 or recall_train == 1.0: continue  # ѵ��������̫��

                                    # if acc_test < 0.80 or recall_test < 0.80: continue  # ���Լ�����̫��
                                    # if acc_test == 1.0 or recall_test == 1.0: continue  # ���Լ�����̫��

                                    if bst_acc_tst < acc_test:
                                        # ���²��Լ���ѵ����������
                                        bst_acc_tra = acc_train
                                        bst_acc_tst = acc_test
                                        bst_recall_tra = recall_train
                                        bst_recall_tst = recall_test
                                        # ����ѵ��������ģ��
                                        fw = open('XGB-output/'+str_col+'/XGB_model.txt', 'wb')
                                        pickle.dump(Xgbc, fw)
                                        fw.close()

                                        test_y = pd.DataFrame(y_test)
                                        test_y['index'] = test_y.index
                                        test_y['XGBԤ��ֵ'] = test_pre
                                        test_y['XGBԤ�����'] = test_proba[:, 1]
                                        test_y.to_csv('XGB-output/'+str_col+'/���Լ�Ԥ����.csv', index=False,encoding='utf_8_sig')

                                        # ��������
                                        score = "ѵ׼��" + str(bst_acc_tra) + '\n'
                                        score = score + "ѵ�٣�" + str(bst_recall_tra) + '\n'
                                        score = score + "��׼��" + str(bst_acc_tst) + '\n'
                                        score = score + "����: " + str(bst_recall_tst)+ '\n'
                                        score = score + "max_depth: " + str(max_depth) + '\n'
                                        # score = score + "booster: " + str(booster) + '\n'
                                        score = score + "learnling-rate: " + str(learning_rate) + '\n'

                                        with open('XGB-output/'+str_col+'/XGB-���ֽ��.txt', 'w') as f:
                                            f.write(score)

                                        # ��������

                                        cfm = confusion_matrix(y_test, test_pre)
                                        print(cfm)
                                        plt.matshow(cfm, cmap=plt.cm.gray)
                                        # plt.savefig('LGIS-output/'+str_col+'/LGIS-��������.png')

                                        labels = ['ˮ��', '�Ͳ�']

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
                                        plt.savefig('XGB-output/' + str_col + '/XGB-��������.png')
                                        plt.close()


                                        print("***XGB*** "+str(bst_acc_tra)+" **** "+str(bst_recall_tra))
                                        print("***XGB*** "+str(bst_acc_tst)+" **** "+str(bst_recall_tst))



