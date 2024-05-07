import pandas as pd
import hydra
from pathlib import Path
import pickle
from zutils import load_emb,load_labels

label_path=Path("A:/ZQJ/ZQJ_CODEreporitory/productMining/embedding_results/allLabels_0118.json")

# 用于一般bert的数据集生成
path_bert_emb=Path("A:\ZQJ\ZQJ_CODEreporitory\OAMINE\OAMine-main\OAMine-main/value_grouping\out_put\iter_7/bert_emb_products")
path_bert_output=Path("A:\ZQJ\ZQJ_CODEreporitory\OAMINE\OAMine-main/test_label\product_type_predict/bert_dataset")



@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
    emb_dir=global_cfg.embedding
    dataset_output=Path(global_cfg.run.labeled_dir)
    path_emb=Path(emb_dir.emb_output_dir)

    print("dataset fine")
    labels=load_labels(label_dir=label_path)
    emb=load_emb(path_dir=path_emb)
    res={"embeddings":[],"category_label": [],"sub_category_label":[]}
    for product_type,embs in emb.items():
        for _,pro_emb in embs.items():
            res["embeddings"].append(pro_emb.tolist())
            res["sub_category_label"].append(product_type)
            res["category_label"].append(labels[product_type][0])
    dataset_df=pd.DataFrame(res)

    print("处理完成")
    dataset_df.to_csv(Path(dataset_output,"datasetLabeled_GPC.csv"))



if __name__ == "__main__":
  main()