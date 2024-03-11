conf="confs/hfavatar-zju/ZJUMOCAP-377-4gpus-infer.conf"
base_exp_dir="exp/CoreView_377_1709621919_slurm_mvs_1_1_1_true"

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 4 \
    --infer_mode odp \
    --novel_pose data/AIST++/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl \
    --novel_pose_type aistplusplus_odp