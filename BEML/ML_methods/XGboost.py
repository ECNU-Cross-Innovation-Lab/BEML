import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import time
from zutils import ML_record,dataset2category,dataset2subcategory



cate_x,cate_y,labels=dataset2category()
subcate_x,subcate_y,subcate_labels=dataset2subcategory()
# print(cate_x.shape)
# print(subcate_x)
# print(type(subcate_x[0]))

train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

values, counts = np.unique(test_label, return_counts=True)
Tvalues, Tcounts = np.unique(train_label, return_counts=True)
art_name=["max_depth"]

cate_recorder=ML_record(attribute_name=["max_depth","n_estimators"])
subcate_recorder=ML_record(attribute_name=["max_depth","n_estimators"])




for dep in [5, 10, 35, 40, 45]:
    for n_estimators in range(50,100,5):
        xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=dep, min_child_weight=1,
                                  subsample=1, colsample_bytree=1, gamma=0.1, reg_alpha=0.01, reg_lambda=3)
        xgb_model.fit(train_data, train_label)

        xgb_model_subcate = xgb.XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=dep, min_child_weight=1,
                                  subsample=1, colsample_bytree=1, gamma=0.1, reg_alpha=0.01, reg_lambda=3)
        xgb_model_subcate.fit(train_subcate_data, train_subcate_label)


        predict_xgb=xgb_model.predict(test_data)
        # predict_subcate_xgb=xgb_model_subcate.predict(test_subcate_data)
        cate_recorder.rec_score(pre_y=predict_xgb,true_y=test_label,label=range(len(labels)),atr={"max_depth":dep,"n_estimators":n_estimators})
        # subcate_recorder.rec_score(pre_y=predict_subcate_xgb,true_y=test_subcate_label,label=subcate_labels,atr={"max_depth":dep,"n_estimators":n_estimators})



cate_recorder.save_record("XGBOOST_GPC")
# subcate_recorder.save_record("subcategory_XGBOOST")
