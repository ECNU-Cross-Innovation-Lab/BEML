
import sys
sys.path.append("..")
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
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



# 记录各项指标
recorder=ML_record(attribute_name=['l1_ratio'])
subcate_recorder=ML_record(attribute_name=['l1_ratio'])

kernals=['linear']



for l1_ratio in range(0,11,2):
        atr = {'l1_ratio': l1_ratio}
        model=ElasticNet(alpha=1,l1_ratio=l1_ratio/10.0,max_iter=5000,random_state=7,)
        model.fit(train_data,train_label)
        predict_y = model.predict(test_data)
        predict_y=[round(predict) for predict in predict_y]

        # print(predict_y[:10])
        recorder.rec_score(pre_y=predict_y, true_y=test_label,label=range(len(labels)),atr=atr)
# recorder.save_record("svm_cate")
recorder.save_record("elasticCate_GPC")


# for l1_ratio in range(0,11,2):
#         atr = {'l1_ratio': l1_ratio }
#         model=ElasticNet(alpha=1,l1_ratio=l1_ratio/10.0,max_iter=5000,random_state=7,)
#         model.fit(train_subcate_data,train_subcate_label)
#         predict_y = model.predict(test_subcate_data)
#         predict_y=[round(predict) for predict in predict_y]
#         subcate_recorder.rec_score(pre_y=predict_y, true_y=test_subcate_label,label=subcate_labels,atr=atr)
# # subcate_recorder.save_record("svm_subcate")
# subcate_recorder.save_record("elastic_subcate")

        # res_dict[str(float(c)/20.0)+'_'+ker+"_predict"]=result

