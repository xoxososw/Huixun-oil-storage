#6RT为例

import pickle
import pandas as pd
import numpy as np
name=['/6RT/ADA-output/ADA_model.pkl',
      '/6RT/GBDT-output/GBDT_model.pkl',
      '/6RT/RF-output/RF_model.pkl',
      '/6RT/XGB-output/XGB_model.pkl'
]
merged_results = []
data = pd.read_csv('./特征提取/pca/小层+测井数据+录井+测试+6RT.csv')
for i in range(4):
    path='./models'+name[i]
    model = pickle.load(file=open(path, 'rb'))
    # 使用模型进行预测
    data1 = data.drop('cellid', axis=1)
    predictions = model.predict(data1)
    predicted_pro = model.predict_proba(data1)
    # print(predicted_pro)
    # 打印预测结果
    print(predictions)
    print(len(predictions))
    print("预测为油田的小层个数:", np.sum(predictions == 1))
    print("预测为水层的小层个数:", np.sum(predictions == 0))
    # 保存测试集上的预测结果
    test_y = pd.DataFrame()
    test_y['index'] = data.index
    test_y['cellid'] = data['cellid']
    yucezhi=name[i].split('/')[2].split('-')[0]+'预测值'
    yucegailv=name[i].split('/')[2].split('-')[0]+'预测概率'
    test_y[yucezhi] = predictions
    test_y[yucegailv] = predicted_pro[:, 1]
    test_y = test_y.sort_values(by=yucegailv, ascending=False)
    test_y.to_csv('./modelce/6RT/' + name[i].split('/')[2] + '/' + '预测结果.csv', index=False, encoding='utf_8_sig')
    merged_results.append(test_y)

# 合并四个结果
merged_result = merged_results[0]
for result in merged_results[1:]:
    merged_result = pd.merge(merged_result, result, on=['index','cellid'])

# 计算四个预测概率的平均值
merged_result['平均预测概率'] = merged_result[['ADA预测概率', 'GBDT预测概率', 'RF预测概率', 'XGB预测概率']].mean(axis=1)

# 按照平均预测概率从高到低排序
sorted_result = merged_result.sort_values(by='平均预测概率', ascending=False)

# 保存排序后的结果
sorted_result.to_csv('./modelce/6RT/merge-result/合并预测结果.csv', index=False, encoding='utf_8_sig')

