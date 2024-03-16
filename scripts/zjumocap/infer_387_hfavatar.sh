conf="confs/hfavatar-zju/ZJUMOCAP-377-4gpus-infer.conf"
base_exp_dir="exp/CoreView_377_1709621919_slurm_mvs_1_1_1_true"

# advanced motion
# --novel_pose data/AIST++/motions/gBR_sFM_cAll_d05_mBR2_ch09.pkl \

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 4 \
    --infer_mode odp \
    --novel_pose data/AIST++/motions/gBR_sBM_cAll_d04_mBR1_ch06.pkl \
    --novel_pose_type aistplusplus_odp