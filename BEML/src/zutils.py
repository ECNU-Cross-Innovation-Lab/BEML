import time
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import distance
import json,csv
import utils
import random
import pandas as pd
from sklearn.metrics import f1_score,recall_score,accuracy_score
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

embedding_dir=Path("emb_products")
label_dir=Path("LabelsOld.json")
final_emb_dir=Path(f"final.emb.pkl") #储存了所有品类在潜在空间中的平均位置表示

# 存放种类预测的所有原始数据的位置
# 存储格式为csv，["embeddings","category_label","subcategory_label"]
category_dataset_path=Path("A:\ZQJ\ZQJ_CODEreporitory\OAMINE\OAMine-main/test_label\product_type_predict/dataset3\dataset_flabel.csv")



class PT_distance():
    rank_method=""
    pt_distance={}#记录了其他type到该pt_name的距离，为pt_name到distance的映射{str:float}
    pt_rank=[] #记录了其他type到该pt_name的升序排列，[str]
    pt_name=""
    def __init__(self, pt_name):
        self.pt_name=pt_name


    #     基于给出的各个type的定位，按照给定的方法计算各个type之间的距离并填入对应的数据结构中
    def cal_distance(self,emb_dist,method):
        self.pt_value=emb_dist[self.pt_name]

        # 计算协方差矩阵
        cov_list=[]
        cov_matrix=None
        inv_cov_matrix=None
        if method=="MAHA":
            for _,pt_value in emb_dist.items():
                cov_list.append(pt_value)
            cov_matrix=np.cov(np.vstack(cov_list),rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        for pt_name,pt_value in emb_dist.items():
            if pt_name != self.pt_name:
                #采用欧式距离
                if method=="EU":
                    self.pt_distance[pt_name]=np.linalg.norm(self.pt_value,pt_value)
                    self.pt_rank.append((pt_name))
                #采用余弦相似度
                elif method=='COSS':
                    dot_product = np.dot(self.pt_value, pt_value)
                    norm_vector1 = np.linalg.norm(self.pt_value)
                    norm_vector2 = np.linalg.norm(pt_value)
                    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
                    self.pt_distance[pt_name]=cosine_similarity
                    self.pt_rank.append(pt_name)
                # 采用余弦距离
                elif method=='COS':
                    dot_product = np.dot(self.pt_value, pt_value)
                    norm_vector1 = np.linalg.norm(self.pt_value)
                    norm_vector2 = np.linalg.norm(pt_value)
                    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
                    self.pt_distance[pt_name] = 1-cosine_similarity
                    self.pt_rank.append(pt_name)
                # 采用马氏距离
                elif method=="MAHA":
                    self.pt_distance[pt_name] =distance.mahalanobis(self.pt_value,pt_value,inv_cov_matrix)
                    self.pt_rank.append(pt_name)
                else:
                    print("该种方法未收录")
        #             升序排序
        self.pt_rank=sorted(self.pt_distance,key=lambda x: self.pt_distance[x],reverse=False)
        # print(pt_rank,"\n")
        # print("FOR",self.pt_name,"first five ranks are",pt_rank[:5])


# 多分类模型结果记录
class ML_record():

    def __init__(self, attribute_name:list):
        self.score_dict = {}  # 记录每次预测的评分，并按照属性进行分组划分
        self.res_dict = {}  # 记录每次预测的结果
        self.attributes = []
        self.score_dict['Weighted precision'] = []
        self.score_dict['Weighted recall'] = []
        self.score_dict['weighted f1-score'] = []
        self.score_dict['Macro precision'] = []
        self.score_dict['Macro recall'] = []
        self.score_dict['Macro f1-score'] = []
        self.score_dict['Micro precision'] = []
        self.score_dict['Micro recall'] = []
        self.score_dict['Micro f1-score'] = []



        self.attributes=attribute_name
        for atr in attribute_name:
            self.score_dict[atr]=[]
        # self.res_dict['true']=y

    # 传入每次预测得到的结果以及对应属性的字典
    def rec_predict(self,pre_y,atr:dict):
        self.score_dict['F1'].append(f1_score(self.res_dict['true'], pre_y, average='weighted'))
        self.score_dict['accuracy'].append(accuracy_score(self.res_dict['true'], pre_y))
        self.score_dict['Recall'].append(recall_score(self.res_dict['true'], pre_y, average="micro"))
        at=[]
        for attribute in self.attributes:
            self.score_dict[attribute].append(atr[attribute])
            at.append(attribute)
        self.res_dict['_'.join(at)]=pre_y

    def rec_score(self,pre_y,true_y,atr:dict,label):

        self.score_dict['Weighted precision'].append( precision_score(true_y, pre_y, average='weighted'))
        self.score_dict['Weighted recall'].append(recall_score(true_y, pre_y,labels=label,average='weighted'))
        self.score_dict['weighted f1-score'].append(f1_score(true_y, pre_y, average='weighted'))
        self.score_dict['Macro precision'].append(precision_score(true_y, pre_y, average='macro'))
        self.score_dict['Macro recall'].append(recall_score(true_y, pre_y,labels=label,  average='macro'))
        self.score_dict['Macro f1-score'].append(f1_score(true_y, pre_y, average='macro'))
        self.score_dict['Micro precision'].append(precision_score(true_y, pre_y, average='micro'))
        self.score_dict['Micro recall'].append(recall_score(true_y, pre_y,labels=label,  average='micro'))
        self.score_dict['Micro f1-score'].append(f1_score(true_y, pre_y, average='micro'))
        print(self.score_dict)

        for attribute in self.attributes:
            self.score_dict[attribute].append(atr[attribute])

    def save_record(self,name):
        path1=Path("predict_result1121",name+"score.csv")
        res_score = pd.DataFrame(self.score_dict)
        res_score.to_csv(path1)

    #        保存最终结果到目标文件
    def save_csv(self,name):
        path1=Path("predict_result",name+"score.csv")
        path2=Path("predict_result",name+"result.csv")
        res_df = pd.DataFrame(self.res_dict)
        res_score = pd.DataFrame(self.score_dict)
        res_df.to_csv(path2)
        res_score.to_csv(path1)

# 判断两个品类是否相同

def is_same_category(pt_1:str,pt_2:str,level,category_dict:dict)->bool:
    # print("\n")
    # print(pt_1,":",category_dict[pt_1][level])
    # print(pt_2,":",category_dict[pt_2][level])
    if category_dict[pt_1][level]==category_dict[pt_2][level]:
        return True
    else:
        return False
# 统计不同品类之间的标签差异

def statis_labels():
    product_label=load_labels()
    LabelInLevel=dict()
    for pro,labels in product_label.items():
        for index,label in enumerate(labels):
            if index in LabelInLevel:
                if label in LabelInLevel[index]:
                    LabelInLevel[index][label]+=1
                else: LabelInLevel[index][label]=1
            else:
                LabelInLevel[index]=dict()
                LabelInLevel[index][label]=1
    # print(LabelInLevel)
    with open('labels_statisticOld.json', 'w') as json_file:
        json.dump(LabelInLevel, json_file, indent=4)
# 返回所有产品编码后的结果，以字典{品类名：{产品名:潜在空间坐标}}形式返回

def load_emb(path_dir=embedding_dir)->dict:
    pts = path_dir
    print("full path:", str(pts))
    pts_emb = dict()
    for item in pts.iterdir():
        if item.is_file() and item.suffix == '.pkl':
            pt_name = item.name[:-len(".emb.pkl")]
            # print("begin loading", pt_name)
            with open(item,'rb') as file:
                products=pickle.load(file)
                pts_emb[pt_name]=products
                # print(products)
    return pts_emb

def load_labels(label_dir=Path("LabelsOld.json"))->dict:
    with open(label_dir,"r") as file:
        data=json.load(file)
    return data

def cal_position():
    pts=embedding_dir
    print("full path:",str(pts))
    pt_emb=dict()
    for item in pts.iterdir():
        if item.is_file() and item.suffix == '.pkl':
            pt_name = item.name[:-len(".emb.pkl")]
            print("begin calculate", pt_name)
            value_lis=[]
            with open(item, 'rb') as file:
                products = pickle.load(file)
                # 根据各个产品的编码定位，生成整个产品品类的定位
                for _,value in products.items():
                    # print(type(value))
                    # print(value.shape)
                    value_lis.append(value)
                pt_value=np.mean(value_lis,axis=0)
                print(pt_name,pt_value.shape)
            pt_emb[pt_name]=pt_value
    utils.Pkl.dump(pt_emb, Path(f"final.emb.pkl"))

def load_embproducts():
    res_dir = Path(embedding_dir,f"final.emb.pkl")
    with open(res_dir,'rb') as file:
        pts=pickle.load(file)
        return pts
# 制作训练集
#  将100个品类表示，任意抽取 一对 作为样本，由模型以及余弦相似度方法输出一个两者概率相等的值，再由两者标签的相似度输出一个概率值
# 可能出现训练集偏置的问题
# 由tranin_test_prop控制训练集和测试集的划分
def make_train_test():
    labels=load_labels()
    x_name=[]
    y=[]
    x=[]
    get_ptname=[]#防止重复
    with open (final_emb_dir,"rb") as file:
        final_emb=pickle.load(file)
        for pt_name,value in final_emb.items():
            # print(type(value))
            get_ptname.append(pt_name)
            for pt_name2,value2 in final_emb.items():
                if pt_name2 not in get_ptname:
                    f_value=value.copy()
                    x_name.append((pt_name,pt_name2))
                    if is_same_category(pt_name,pt_name2,0,labels):
                        y.append(1)
                    else:y.append(-1)
                    f_value=np.concatenate((f_value,value2))
                    x.append(f_value)
        res_x=np.array(x)
        res_y = np.array(y)
        # print(res_x)
        # print(res_x.shape)
        # print(res_y[:5])
    return x_name,res_x,res_y

def make_train_test_products():
    labels = load_labels()
    path_emb=Path("emb_products")
    final_emb={}
    final_emb_name=[]
    get_ptname=[]

    #     TODO
    # 制作数据集，每个数据集取100个点和其他的进行随机搭配组合
    # 需要逐行写入制作数据集
    #
    random.seed(7)
    for emb in path_emb.iterdir():
        # print(emb.name[:-8])
        final_emb[emb.name[:-8]]={}
        final_emb_name.append(emb.name[:-8])
        with open(emb,'rb') as file:
            embs=pickle.load(file)

            samples=random.sample(embs.items(),10)
            final_emb[emb.name[:-8]]=dict(samples)


    f_x=open("data_x.csv","w",newline='')
    f_y = open("data_y.csv", "w",newline='')
    writerx = csv.writer(f_x)
    writery = csv.writer(f_y)
    get_pts=[]
    start_time = time.time()

    for pt_name, value in final_emb.items():
            print("begin:",pt_name)
            for product,emb in value.items():
                for pt_name2, value2 in final_emb.items():
                    # print("组合：",pt_name2)
                    if pt_name2 in get_pts:continue
                    for product2,emb2 in value2.items():
                        if product2==product:continue
                        else:
                            f_value = emb.copy()
                            f_value = np.concatenate((f_value, emb2))
                            writerx.writerow(f_value)
                            if is_same_category(pt_name, pt_name2, 0, labels):
                                writery.writerow([1])
                            else:
                                writery.writerow([-1])
            get_pts.append(pt_name)

    end_time=time.time()
    print("运行时间为%s min"%((end_time-start_time)/60))

    f_x.close()
    f_y.close()

# 返回训练特征和对应标签
# 需要将字符串映射为ndarray
def dataset2category(dataset_path=category_dataset_path):
    df=pd.read_csv(category_dataset_path)
    ser_fea=df["embeddings"].str.replace('\n', '')
    ser_fea=ser_fea.to_numpy()
    x=[]
    for fea in ser_fea:
        x.append(np.array(json.loads(fea)))
    # print("x:",type(x[0]))

    ser_label=df["category_label"].str.replace('\n', '')
    ser_label,unique=ser_label.factorize()
    label_int = [i for i in range(len(unique))]
    return x,ser_label,label_int


def dataset2subcategory(dataset_path=category_dataset_path):
    df = pd.read_csv(category_dataset_path)

    ser_fea = df["embeddings"].str.replace('\n', '')
    x = []
    for fea in ser_fea:
        x.append(np.array(json.loads(fea)))
    print("x:", type(x[0]))
    print(len(x))

    ser_label = df["sub_category_label"]
    ser_label, unique = ser_label.factorize()
    ser_fea = ser_fea.to_numpy()

    print(ser_fea.shape)
    print("标签个数为", len(unique))
    # print(unique)
    label_int=[i for i in range(len(unique))]
    # print(len(label_int))
    return x, ser_label,label_int




if __name__=='__main__':
    dataset2subcategory()