python3 -u -m light.pytorch.launch --master_port 666 --nproc_per_node 8 train.py \
--work_dir "/mnt/group-ai-medical-sz/private/jinxixiang/results/work_dir4/config45_PatchGCN_lymph_10x" \
--config './configs/config45_PatchGCN_lymph_10x.yaml' \
--log_file 'log_train.txt'