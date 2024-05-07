import logging
import math
import random
from pathlib import Path
from tqdm import tqdm
import hydra
import torch
from sentence_transformers import losses
from sentence_transformers.models import Transformer
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from contextualized_sbert import (load_multitask_model,
                                  preprocess_singleton_data,
                                  preprocess_singleproduct_data)
import utils
from contextualized_sbert.data import (load_binary_dataset, load_clf_dataset,
                                       load_triplet_dataset)
from contextualized_sbert.models import (ClassifierClfLoss,
                                         ContextualizedBinaryClfEvaluator,
                                         EntityPooling, EntitySBERT)

candidate_bert_path=Path("A:\ZQJ\ZQJ_CODEreporitory\OAMINE\OAMine-main\OAMine-main\data/amazon/normalBert_candidate_sample")


#### Code to print debug information to stdout
utils.Log.config_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


# 进行初始数据的映射
@hydra.main(config_path="../exp_config", config_name="config")
def main(global_cfg):
    cfg_dir = global_cfg.embedding
    cfg_bert = global_cfg.tuning
    all_examples=[]
    pts=[]
    pt2range=dict()
    pt2pn=dict()
    transformer = Transformer(cfg_dir.pretrained_model)
    pooling = EntityPooling(transformer.get_word_embedding_dimension())
    model = EntitySBERT(modules=[transformer, pooling])
    model.max_seq_length = cfg_bert.max_seq_length
    candidate_path=Path(cfg_dir.candidate_dir)
    logger.info(candidate_path.name)


    for candidate_file in candidate_path.glob("*.asin.jsonl"):
            pt_name = candidate_file.stem[:-len(".asin")]
            pts.append(pt_name)
    for pt in tqdm(pts):
        docs = utils.JsonL.load(Path(candidate_path, f"{pt}.jsonl"))

        pt2pn[pt] = docs
        examples=preprocess_singleproduct_data(docs, model.tokenizer, max_seq_length=model.max_seq_length,
                                         disable_tqdm=True)
        start = len(all_examples)
        all_examples.extend(examples)
        end = len(all_examples)
        pt2range[pt]=(start,end)
    logger.info(f"{len(all_examples)} examples")

    def report_progress(progress: float):
        print(f"{progress * 100:.2f}%")

    all_embeddings = model.encode(all_examples, batch_size=cfg_bert.batch_size, show_progress_bar=False,
                                  progress_callback=report_progress)

    for pt in pts:
        start, end = pt2range[pt]
        # phrase2context_idx = pt2phrase_context_map[pt]
        pn = pt2pn[pt]
        pt_embeddings = all_embeddings[start:end]
        print("start:end", start, ":", end)
        print("pt_embeddings:", type(pt_embeddings))
        print(pt_embeddings.shape)
        # print("product_name",type(pn))
        static_embeddings = dict()
        print("length of pn is", len(pn))
        # TODO

        for i in range(len(pn)):
            if pn[i] in static_embeddings.keys():
                print(pn[i], "has in dict")
            static_embeddings[pn[i]] = pt_embeddings[i]
            # print(pn[i],"refer to",pt_embeddings[i])
        print("length of dict is", len(static_embeddings))
        utils.Pkl.dump(static_embeddings, Path(cfg_dir.emb_output_dir, f"{pt}.emb.pkl"))
        # print("document is loaded to：",embedding_output_path, f"{pt}.emb.pkl")

if __name__ == "__main__":
  main()
