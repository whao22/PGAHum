# conf="confs/hfavatar-people_snapshot/PeopleSnapshot-female-3-casual-mono-4gpus.conf"
conf="confs/hfavatar-people_snapshot/PeopleSnapshot-female-3-casual-mono-4gpus.conf"
base_exp_dir="exp/Peoplesnapshot-female-3-casual_1711463792_slurm_mono_1_1_3_true"

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 2 \
    --infer_mode nvs \
    --novel_pose data/AIST++/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl \
    --novel_pose_type aistplusplus_odp