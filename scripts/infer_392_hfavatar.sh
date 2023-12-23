exp_name="CoreView_377_1702989327_slurm_w/_inner-w/_init_sdf-w/_tri"
base_exp_dir="exp/${exp_name}"

python validate.py confs/hfavatar-zju/ZJUMOCAP-377-mono-4gpus.conf \
    --base_exp_dir ${base_exp_dir} \
    --novel-view
    