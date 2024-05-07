
import hydra
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split

from zutils import ML_record,dataset2category,dataset2subcategory,save_model

def main():
    cate_x,cate_y,labels=dataset2category()
    # subcate_x,subcate_y,subcate_labels=dataset2subcategory()

    train_data,test_data,train_label,test_label=train_test_split(cate_x,cate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)
    # train_subcate_data,test_subcate_data,train_subcate_label,test_subcate_label=train_test_split(subcate_x,subcate_y,random_state=1,train_size=0.7,test_size=0.3,shuffle=True)

    values, counts = np.unique(test_label, return_counts=True)
    Tvalues, Tcounts = np.unique(train_label, return_counts=True)

    # 记录各项指标
    recorder=ML_record(attribute_name=['c','kernal'])
    subcate_recorder=ML_record(attribute_name=['c','kernal'])
    kernals=['linear', 'poly', 'rbf', 'sigmoid']
    # kernals=['rbf']


    for c in range(10,20,2):
        for ker in kernals:
            atr = {'kernal': ker, 'c': c}
            print(atr)
            model = svm.SVC(C=float(c)/20.0,kernel=ker,decision_function_shape='ovr',probability=True)
            model.fit(train_data,train_label)
            save_model("SVM_rbf_14",model)
            predict_y = model.predict(test_data)
            recorder.rec_score(pre_y=predict_y, true_y=test_label,label=range(len(labels)),atr=atr)
    # recorder.save_record2dir("SVM_rbf_14","../model_saved")
    recorder.save_record("svm_GPC")
    # for c in range(10,20,2):
    #     for ker in kernals:
    #         atr = {'kernal': ker, 'c': c}
    #         model = svm.SVC(C=float(c)/20.0,kernel=ker,decision_function_shape='ovr',probability=True,random_state=7)
    #         model.fit(train_subcate_data,train_subcate_label)
    #         predict_y = model.predict(test_subcate_data)
    #         subcate_recorder.rec_score(pre_y=predict_y, true_y=test_subcate_label,label=subcate_labels,atr=atr)
    # # subcate_recorder.save_record("svm_subcate")
    # subcate_recorder.save_record("svm_bert_subcate")

    # res_dict[str(float(c)/20.0)+'_'+ker+"_predict"]=result


if __name__=='__main__':
    main()