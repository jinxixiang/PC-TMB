python3 -u -m light.pytorch.launch --master_port 666 --nproc_per_node 8 train.py \
--work_dir "." \
--config './configs/config01_PatchGCN_7cancer_10x3247_balanced.yaml' \
--log_file 'log_train.txt'