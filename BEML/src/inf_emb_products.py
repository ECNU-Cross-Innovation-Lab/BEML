import logging
import random
from pathlib import Path
from typing import List
from tqdm import tqdm
import hydra
# import ray
import numpy as np
# from ray import tune
import utils
from contextualized_sbert import (load_multitask_model,
                                  preprocess_singleton_data,
                                  preprocess_singleproduct_data)
from contextualized_sbert.data_utils import match_context
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

logger = logging.getLogger(__name__)


# zqj的embedding,将单个商品的信息进行编码，生成编码后的结果
def run_embedding_on_pts(config):
  print("start embedding for PTS")

  cfg = config["cfg"]
  pts: List[str] = config["pts"]

  model_dir = Path(cfg.model_dir)
  output_dir = Path(cfg.sample_output_dir)
  random.seed(cfg.rnd_seed)

  utils.IO.ensure_dir(output_dir)
  model = load_multitask_model(model_dir, "embedding")
  print("model is in",model_dir)
  # load data from multiple PTs and combine them into
  # one large dataset for inference
  logger.info("Prep")

  pt2range = dict()
  all_examples = []
  # 记录每一个product type对应的名字
  pt2pn=dict()

  for pt in tqdm(pts):

    # 载入A:\ZQJ\ZQJ_CODEreporitory\OAMINE\OAMine-main\OAMine-main\data\amazon\candidate 中的数据
    # TODO Z1
    # 基于文件名对进行编码的商品进行筛选 进行筛选
    # docs = utils.JsonL.load(Path(cfg.candidate_dir, f"{pt}.chunk.jsonl"))


    #   使用分组后的数据预测
    docs = utils.JsonL.load(Path(cfg.sample_candidate_dir, f"{pt}.chunk.jsonl"))
    all_phrases = [p for doc in docs for p in doc]
    # sort by phrase popularity,
    # 对属性的出现次数按照频率进行排列
    # unique_phrases = utils.Sort.unique_by_frequency(all_phrases)

    # map phrases to their contexts w/ sampling
    # phrase2context_idx, contexts = match_context(unique_phrases, docs, sampling=cfg.sampling)
    # contexts=[str] 包含每一个产品描述的列表,但此处未用index来记录具体索引

    contexts = [" ".join(doc) for doc in docs]
    pt2pn[pt]=contexts

    # preprocess data
    # 返回了对应的tokenizer化后和对应补充信息的数据形式
    #     dataset.append({
    #         "input_ids": input_ids, #tokenizer化后的结果
    #         "token_type_ids": token_type_ids, # 全0
    #         "attention_mask": attention_mask, #全1
    #         "pooling_mask": pooling_mask, #标识出对应entity的pooling_mask
    #     })

    # print("contexts:",len(contexts))

    examples = preprocess_singleproduct_data(contexts, model.tokenizer, max_seq_length=model.max_seq_length,
                                         disable_tqdm=True)

    start = len(all_examples)
    all_examples.extend(examples)
    end = len(all_examples)
    # logger.info(f"{pt} {end - start} examples")

    pt2range[pt] = (start, end)

  # run BERT inference
  logger.info(f"{len(all_examples)} examples")

  # print(len(all_examples))
  def report_progress(progress: float):
    print(f"{progress * 100:.2f}%")

  all_embeddings = model.encode(all_examples, batch_size=cfg.batch_size, show_progress_bar=False,
                                progress_callback=report_progress)
  print("Finish encoding")
  # gather embeddings per PT
  logger.info("gather")
  for pt in pts:
    start, end = pt2range[pt]
    # phrase2context_idx = pt2phrase_context_map[pt]
    pn=pt2pn[pt]
    pt_embeddings = all_embeddings[start:end]
    print("start:end",start,":",end)
    print("pt_embeddings:",type(pt_embeddings))
    print(pt_embeddings.shape)
    # print("product_name",type(pn))
    static_embeddings = dict()
    print("length of pn is",len(pn))
    # TODO

    for i in range(len(pn)):
      if pn[i] in static_embeddings.keys():
        print(pn[i],"has in dict")
      static_embeddings[pn[i]]=pt_embeddings[i]
      # print(pn[i],"refer to",pt_embeddings[i])
    print("length of dict is",len(static_embeddings))
    utils.Pkl.dump(static_embeddings, Path(output_dir, f"{pt}.emb.pkl"))
    print("document is loaded to：",output_dir,f"{pt}.emb.pkl")


@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
  cfg = global_cfg.inference.embedding


  # of using all PTs
  # 选择哪些PT进行编码
  if cfg.selected_pt:
    pts = [s.strip() for s in cfg.selected_pt.split(",")]
  else:
    pts = []
    for candidate_file in Path(cfg.candidate_dir).glob("*.chunk.jsonl"):
      pt_name = candidate_file.stem[:-len(".chunk")]
      pts.append(pt_name)
  logger.info(f"Loaded {len(pts)} PTs")
  logger.info(f"First 3 PTs: {pts[:3]}")
  logger.info(f"n_gpu = {cfg.n_gpu}")
  logger.info(f"Group PTs into {cfg.n_gpu} groups")

  grouped_pts= [a.tolist() for a in np.array_split(pts, cfg.n_gpu)]
  # tune.run(run_inference_on_pts, config={
  #   "cfg": cfg,
  #   "pts": tune.grid_search(grouped_pts)
  # }, resources_per_trial={"cpu": cfg.cpu_per_job, "gpu": 1})
  config = {
    "cfg": cfg,
    "pts": pts,
  }
  # 对商品的属性进行编码
  run_embedding_on_pts(config)
  # run_embedding_on_pts(config)


if __name__ == "__main__":
  main()