exp_name="CoreView_377_1707616975_slurm_mvs_w/_inner-w/_init_sdf-w/_patch-rgb50-lpip5-eik0.1"
conf="confs/hfavatar-zju/ZJUMOCAP-377-mono-4gpus.conf"
base_exp_dir="exp/${exp_name}"

python infer.py ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 1 \
    --infer_mode test \
    --novel_view 0 \
    --novel_pose data/data_prepared/CoreView_387