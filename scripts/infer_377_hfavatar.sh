exp_name="CoreView_377_1709621919_slurm_mvs_1_1_1_true"
conf="confs/hfavatar-zju/ZJUMOCAP-377-4gpus.conf"
base_exp_dir="exp/${exp_name}"

python infer.py --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 1 \
    --infer_mode val \
    --novel_view -1 \
    --novel_pose data/data_prepared/CoreView_387