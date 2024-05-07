import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from zutils import ML_record,dataset2category,dataset2subcategory
from sklearn.tree import DecisionTreeClassifier


cate_x,cate_y,labels=dataset2category()
subcate_x,subcate_y,subcate_labels=dataset2subcategory()
# print(cate_x.shape)
# print(subcate_x)
# print(type(subcate_x[0]))

train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

values, counts = np.unique(test_label, return_counts=True)
Tvalues, Tcounts = np.unique(train_label, return_counts=True)
# print("测试集标签分类数为",values)
# print(counts)
# print("训练集标签分类数为",Tvalues)
# print(Tcounts)
# ————————————————————————————进行预测————————————————————————————

atr_name=['max_depth','criterion']
criterion=['gini','entropy']

recorder=ML_record(attribute_name=atr_name)
subcate_recorder=ML_record(attribute_name=atr_name)

print("begin train ")
for cri in criterion:
    for dep in range (5,46,5):
        clf = DecisionTreeClassifier(criterion=cri,max_depth=dep,random_state=7)# 实例化模型，添加criterion参数
        clf = clf.fit(train_data, train_label)# 使用实例化好的模型进行拟合操作
        # clf_subcate = DecisionTreeClassifier(criterion=cri,max_depth=dep,random_state=7)#
        # clf_subcate.fit(train_subcate_data,train_subcate_label)# 使用实例化好的模型进行拟合操作

        predict_dtc=clf.predict(test_data)
#         predict_y=clf_subcate.predict(test_subcate_data)
        recorder.rec_score(pre_y=predict_dtc,true_y=test_label, label=range(len(labels)),atr={'max_depth':dep,'criterion':cri})
#         subcate_recorder.rec_score(pre_y=predict_y,true_y=test_subcate_label,label=subcate_labels,atr={'max_depth':dep,'criterion':cri})
# subcate_recorder.save_record("subcategory_decision_tree")
recorder.save_record("decisionTree_GPC")