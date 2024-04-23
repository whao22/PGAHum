conf="confs/hfavatar-zjumocap/ZJUMOCAP-377-4gpus.conf"
base_exp_dir="exp/CoreView_377_1709621919_slurm_mvs_1_1_1_true"

echo ${base_exp_dir}
# advanced motion
# data/AIST++/motions/gBR_sFM_cAll_d05_mBR2_ch09.pkl

# basic motion
# data/AIST++/motions/gBR_sBM_cAll_d04_mBR1_ch06.pkl

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 2 \
    --infer_mode unseen \
    --novel_pose data/AIST++/motions/gBR_sFM_cAll_d05_mBR2_ch09.pkl \
    --novel_pose_type aistplusplus_odp