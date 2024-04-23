conf="confs/hfavatar-people_snapshot/PeopleSnapshot-male-4-casual-mono-4gpus.conf"
base_exp_dir="exp/CoreView_male-4-casual_1709841604_run_mono_1_1_3_true"

python infer.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --resolution_level 2 \
    --infer_mode nvs \
    --novel_pose data/AIST++/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl \
    --novel_pose_type aistplusplus_odp