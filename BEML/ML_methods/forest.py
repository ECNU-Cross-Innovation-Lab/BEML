from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from zutils import ML_record,dataset2category,dataset2subcategory


cate_x,cate_y,labels=dataset2category()
subcate_x,subcate_y,subcate_labels=dataset2subcategory()
train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
values, counts = np.unique(test_label, return_counts=True)
Tvalues, Tcounts = np.unique(train_label, return_counts=True)

recorder=ML_record(attribute_name=['max_depth','n_estimator'])
subcate_recorder=ML_record(attribute_name=['max_depth','n_estimator'])


for max_depth in range(5,50,5):
    for n_estimator in range (50,100,10):
        atr={'max_depth':str(max_depth),'n_estimator':str(n_estimator)}
        clf = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth,random_state=7)
        clf.fit(train_data,train_label)
        predict_y=clf.predict(test_data)
        recorder.rec_score(pre_y=predict_y,true_y=test_label,label=range(len(labels)),atr=atr)
recorder.save_record("FORESTCate_GPC")
#
# for max_depth in range(5,50,5):
#     for n_estimator in range (50,100,5):
#         atr={'max_depth':str(max_depth),'n_estimator':str(n_estimator)}
#         clf = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth,random_state=7)
#         clf.fit(train_subcate_data,train_subcate_label)
#         predict_y=clf.predict(test_subcate_data)
#         subcate_recorder.rec_score(pre_y=predict_y,true_y=test_subcate_label,label=subcate_labels,atr=atr)
# subcate_recorder.save_record("FOREST_subcate")
