import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from zutils import ML_record,dataset2category,dataset2subcategory,save_model


cate_x,cate_y,labels=dataset2category()
subcate_x,subcate_y,subcate_labels=dataset2subcategory()
# print(cate_x.shape)
# print(subcate_x)
# print(type(subcate_x[0]))

train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

values, counts = np.unique(test_label, return_counts=True)
Tvalues, Tcounts = np.unique(train_label, return_counts=True)

# ————————————————————————————进行预测————————————————————————————

activation=['identity', 'logistic', 'tanh', 'relu']
# activation=[ 'relu']
solver=['lbfgs', 'sgd', 'adam']
# solver=[ 'adam']
cate_recorder=ML_record(attribute_name=['activation','solver'])
subcate_recorder=ML_record(attribute_name=['activation','solver'])



for act in activation:
    for sol in solver:
        atr={'activation':act,'solver':sol}
        clf = MLPClassifier(solver=sol,activation=act, alpha=1e-5,hidden_layer_sizes=(100,50,20), random_state=7,max_iter=10000)
        clf.fit(train_data,train_label)
        predict_y=clf.predict(test_data)
        cate_recorder.rec_score(pre_y=predict_y,true_y=test_label,label=range(len(labels)),atr=atr)
        # save_model("MLP_relu_adam",clf)
        # subcate_clf = MLPClassifier(solver=sol,activation=act, alpha=1e-5,hidden_layer_sizes=(100,50,20), random_state=7,max_iter=100000,)
        # subcate_clf.fit(train_subcate_data,train_subcate_label)

        # predict_subcate=subcate_clf.predict(test_subcate_data)
        # subcate_recorder.rec_score(pre_y=predict_subcate,true_y=test_subcate_label,label=subcate_labels,atr=atr)
cate_recorder.save_record("categoryMLP_GPC")
# cate_recorder.save_record2dir("MLP_relu_adam","../model_saved")
