conf="confs/hfavatar-synthetic_human/SyntheticHuman-megan-mono-4gpus.conf"
base_exp_dir="exp/SyntheticHuman-megan_1711177582_slurm_mono_1_1_1_true"

# advanced motion
# data/AIST++/motions/gBR_sFM_cAll_d05_mBR2_ch09.pkl

# basic motion
# data/AIST++/motions/gBR_sBM_cAll_d04_mBR1_ch06.pkl

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 4 \
    --infer_mode nvs \
    --novel_pose data/AIST++/motions/gBR_sFM_cAll_d05_mBR2_ch09.pkl \
    --novel_pose_type aistplusplus_odp