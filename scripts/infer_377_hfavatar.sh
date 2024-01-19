
# exp_name="CoreView_377_1704281050_slurm_w/_inner-w/_init_sdf-w/_patch-rgb50-lpip5-eik0.1"
exp_name="CoreView_377_1704374745_slurm_w/_inner-w/_init_sdf-w/_patch-rgb50-lpip5-eik0.1"
conf="confs/hfavatar-zju/ZJUMOCAP-377-4gpus.conf"
base_exp_dir="exp/${exp_name}"

python infer.py ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 1 \
    --infer_mode val \
    # --novel_view -1 
    # --novel_pose data/data_prepared/CoreView_392