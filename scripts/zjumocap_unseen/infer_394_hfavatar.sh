conf="confs/hfavatar-zjumocap/ZJUMOCAP-394-4gpus.conf"
base_exp_dir="exp/CoreView_394_1710683923_slurm_mvs_1_1_3_true"

echo ${base_exp_dir}
# advanced motion
# --novel_pose data/AIST++/motions/gBR_sFM_cAll_d05_mBR2_ch09.pkl \

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 2 \
    --infer_mode unseen \
    --novel_pose data/AIST++/motions/gBR_sBM_cAll_d04_mBR1_ch06.pkl \
    --novel_pose_type aistplusplus_odp