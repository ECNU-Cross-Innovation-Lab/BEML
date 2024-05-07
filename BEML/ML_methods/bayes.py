from sklearn.naive_bayes import MultinomialNB,ComplementNB,GaussianNB
from sklearn.linear_model import BayesianRidge,ARDRegression
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from zutils import ML_record,dataset2category,dataset2subcategory
from sklearn.tree import DecisionTreeClassifier

cate_x,cate_y,labels=dataset2category()
subcate_x,subcate_y,subcate_labels=dataset2subcategory()

train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
# train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

values, counts = np.unique(test_label, return_counts=True)
Tvalues, Tcounts = np.unique(train_label, return_counts=True)



# 记录各项指标
recorder=ML_record(attribute_name=['model_type'])
subcate_recorder=ML_record(attribute_name=['model_type'])

kernals=['model_type']
function={'Gaussian':GaussianNB,'Multinomial':MultinomialNB,"Complement":ComplementNB}
# function={'Gaussian':GaussianNB,'BayesRidge':BayesianRidge,'ARD_Bayes':ARDRegression}
# function={'BayesRidge':BayesianRidge,'ARD_Bayes':ARDRegression}
# function={'Gaussian':GaussianNB}

for model,func in function.items():
        atr = {'model_type':model}
        model=GaussianNB()
        model.fit(train_data,train_label)
        predict_y = model.predict(test_data)
        predict_y=[round(predict) for predict in predict_y]

        # model_subcate = func()
        # model_subcate.fit(train_subcate_data, train_subcate_label)
        # predict_y_subcate = model_subcate.predict(test_subcate_data)
        # predict_y_subcate = [round(predict) for predict in predict_y_subcate]
        # subcate_recorder.rec_score(pre_y=predict_y_subcate, true_y=test_subcate_label, label=subcate_labels, atr=atr)
        recorder.rec_score(pre_y=predict_y, true_y=test_label,label=range(len(labels)),atr=atr)
recorder.save_record("naiveBayes_GPC")

# subcate_recorder.save_record("naiveBayes_subcate")
