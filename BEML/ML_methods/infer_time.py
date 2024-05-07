import time
import numpy as np
import pandas as pd
from zutils import ML_record,dataset2category,dataset2subcategory,save_model
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier




cate_x,cate_y,labels=dataset2category()
train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

values, counts = np.unique(test_label, return_counts=True)


xgbmodel=xgb.XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=10, min_child_weight=1)
MLPmodel = MLPClassifier(solver='adam', activation='linear', alpha=1e-5, hidden_layer_sizes=(100, 50, 20), random_state=7,max_iter=10000)
SVMmodel = SVC(C=14.0/ 20.0, kernel='linear', decision_function_shape='ovr', probability=True)
SVMmodel2 = SVC(C=14.0/ 20.0, kernel='poly', decision_function_shape='ovr', probability=True)

# xgbmodel.fit(train_data, train_label)
# MLPmodel.fit(train_data,train_label)
SVMmodel.fit(train_data,train_label)
SVMmodel2.fit(train_data,train_label)
model_list=[SVMmodel,SVMmodel2]

# time_dict={'time':[],'model':['xgbmodel','MLPmodel','SVMmodel']}
time_dict={'time':[],'model':['SVMmodel_linear','svm_poly']}


for model in model_list:
    start = time.time()
    predict_y=model.predict(test_data)
    end=time.time()
    time_dict['time'].append(time.time()-start)
rec_score=pd.DataFrame(time_dict)
rec_score.to_csv('time_svm.csv')


