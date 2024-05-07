TRAIN_GPU=0  # training on single gpu only
INFERENCE_GPU=0 # inference can be run on multiple gpu, each gpu runs inference for one PT at a time
N_INF_GPU=$(echo $INFERENCE_GPU | awk -F',' '{print NF}')  # number of comma + 1

RND_SEED=1

## SET YOUR DATASET PATH IN exp_config/dataset/your_data.yaml
## SET PRETRAINED MODEL IN exp_config/tuning/multitask.yaml
#EXP_DIR=A:/ZQJ/papers/WWW/OA-MineOpen-World Attribute Mining for E-Commerce/OAMine-main/OAMine-main/value_grouping/output # where outputs will be saved
EXP_DIR=A:/ZQJ/ZQJ_CODEreporitory/productMining/embedding_results/embedding_results0328
EVAL_DIR=A:/ZQJ/ZQJ_CODEreporitory/OAMINE/OAMine-main/OAMine-main/data/amazon/test   # ground truth evaluation clusters, either dev or test set
TRAIN_ITER=7
# CLF_INF=false  # whether to use classifier inference

USE_WANDB=false
# uncomment and set up wandb to use wandb
# export WANDB_API_KEY=2d01e86a26931a690d466139632cfdbc6c64f9eb
# export WANDB_PROJECT=oamine-release
# export WANDB_TAGS=release
# # export WANDB_MODE=offline
# USE_WANDB=true


for ((ITER=7;ITER<=$TRAIN_ITER;ITER++));
do
#  PREV_ITER=$(($ITER-1))
#  if (($ITER>1));
#  then
#    PREV_ITER_INF_DIR=$EXP_DIR/iter_$PREV_ITER/ensemble_inf
#  else
#    PREV_ITER_INF_DIR=null
#  fi
#
#  if [ "$ITER" == "$TRAIN_ITER" ]; then
#    CLF_INF=true
#  else
#    CLF_INF=false
#  fi

#echo =======================================================
#echo ============ Fine-tuning Data Generation ==============
#echo =======================================================
#
#  python src/data_gen/gen_binary.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED preprocessing.binary.inference_dir=$PREV_ITER_INF_DIR
#  python src/data_gen/gen_triplet.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED preprocessing.triplet.inference_dir=$PREV_ITER_INF_DIR
#  python src/data_gen/gen_clf.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED preprocessing.clf.inference_dir=$PREV_ITER_INF_DIR
#  python src/data_gen/gen_multitask.py run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
#
#


#
#  echo =======================================================
#  echo =============== Multitask Fine-tuning =================
#  echo =======================================================
#  python src/sbert_multitask.py run.train_gpu=$TRAIN_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
#
#
#
#  echo =======================================================
#  echo ===================== Inference =======================
#  echo =======================================================
#
##此处的emb操作实际上在下面的inf_ensemble_dist处有重合
#  CUDA_VISIBLE_DEVICES=$INFERENCE_GPU python src/inf_emb.py run.n_inf_gpu=$N_INF_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
#  if [ "$CLF_INF" = true ]; then
#    echo "Use classifier inference"
#    # by default classifier inference run on selected PTs (w/ gold labels)
#    # to run full inference, set inference=full (very time consuming due to slow classifier inference)
#    CUDA_VISIBLE_DEVICES=$INFERENCE_GPU python src/inf_clf_dist.py run.n_inf_gpu=$N_INF_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
#    python src/inf_ensemble_dist.py inference=selected_full run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
#  else
#    python src/inf_ensemble_dist.py inference=dbscan run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
#  fi
#
#  echo "Running evaluation"
#  if [ "$USE_WANDB" = true ]; then
#    python src/eval_clustering.py $EXP_DIR/iter_$ITER/ensemble_inf $EVAL_DIR "exp note" true
#  else
#    python src/eval_clustering.py $EXP_DIR/iter_$ITER/ensemble_inf $EVAL_DIR
#  fi

#  最后进行
#    CUDA_VISIBLE_DEVICES=$INFERENCE_GPU python src/inf_emb_products.py run.n_inf_gpu=$N_INF_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
    CUDA_VISIBLE_DEVICES=$INFERENCE_GPU python src/normal_bert_emb.py run.n_inf_gpu=$N_INF_GPU run.exp_dir=$EXP_DIR run.iter_num=$ITER run.rnd_seed=$RND_SEED
done

#训练结束后，进行一次embedding