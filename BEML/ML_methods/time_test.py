import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import time
from zutils import ML_record,dataset2category,dataset2subcategory
from sklearn.neural_network import MLPClassifier


cate_x,cate_y,cate_labels=dataset2category()
subcate_x,subcate_y,subcate_labels=dataset2subcategory()
# print(cate_x.shape)
# print(subcate_x)
# print(type(subcate_x[0]))

start_time_xgb=time.time()
train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
xgb_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=90, max_depth=20, min_child_weight=1,
                                  subsample=1, colsample_bytree=1, gamma=0.1, reg_alpha=0.01, reg_lambda=3)
xgb_model.fit(train_data, train_label)

end_time_xgb=time.time()

print("xgbtime:",end_time_xgb-start_time_xgb)

start_time_mlp=time.time()

subcate_clf = MLPClassifier(solver='sgd',activation='identity', alpha=1e-5,hidden_layer_sizes=(100,50,20), random_state=7,max_iter=10000,)
subcate_clf.fit(train_data,train_label)

end_time_mlp=time.time()
print("xgbtime:",end_time_mlp-start_time_mlp)
