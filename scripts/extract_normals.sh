cd /home/wanghao/WORKSPACE/wanghao/hf-avatar


# 377
python extract_geometry.py --device cuda:1 --conf confs/hfavatar-zjumocap/ZJUMOCAP-377-4gpus.conf --base_exp_dir exp/CoreView_377_1709621919_slurm_mvs_1_1_1_true > 377.log 2>&1

# # 394
# python extract_geometry.py --conf confs/hfavatar-zjumocap/ZJUMOCAP-394-4gpus.conf --base_exp_dir exp/CoreView_394_1710683923_slurm_mvs_1_1_3_true

# 387
python extract_geometry.py --device cuda:1 --conf confs/hfavatar-zjumocap/ZJUMOCAP-387-4gpus.conf --base_exp_dir exp/CoreView_387 > 387.log 2>&1

# # 386
# python extract_geometry.py --conf confs/hfavatar-zjumocap/ZJUMOCAP-386-4gpus.conf --base_exp_dir exp/CoreView_386_1711290959_slurm_mvs_1_1_3 > 386.log 2>&1

# # 393
# python extract_geometry.py --device cuda:2 --conf confs/hfavatar-zjumocap/ZJUMOCAP-393-4gpus.conf --base_exp_dir exp/CoreView_393_1710654406_slurm_mvs_1_1_3_true_woinner > 393.log 2>&1

# # male-3
# python extract_geometry.py --device cuda:3 --conf confs/hfavatar-people_snapshot/PeopleSnapshot-male-3-casual-mono-4gpus.conf --base_exp_dir exp/Peoplesnapshot-male-3-casual_1711424734_slurm_mono_1_1_3_true > male-3.log 2>&1

# # male-4
# python extract_geometry.py --device cuda:3 --conf confs/hfavatar-people_snapshot/PeopleSnapshot-male-4-casual-mono-4gpus.conf --base_exp_dir exp/CoreView_male-4-casual_1709841604_run_mono_1_1_3_true > male-4.log 2>&1

# # megan
# python extract_geometry.py --device cuda:1 --conf confs/hfavatar-synthetic_human/SyntheticHuman-megan-mono-4gpus.conf --base_exp_dir exp/SyntheticHuman-megan_1711268122_slurm_mono_1_1_1_true > megan.log 2>&1

# # jody
# python extract_geometry.py --conf confs/hfavatar-synthetic_human/SyntheticHuman-megan-mono-4gpus.conf --base_exp_dir exp/SyntheticHuman-megan_1712480657_slurm_mono_1_1_1_true

# # male_outfit1
# python extract_geometry.py --conf confs/hfavatar-selfrecon_synthesis/SelfreconSynthesis-male-outfit1-mono-4gpus.conf --base_exp_dir exp/SelfreconSynthesis-male_outfit1_1712047550_slurm_mono_1_1_3_true

# # male_outfit2
# python extract_geometry.py --conf confs/hfavatar-selfrecon_synthesis/SelfreconSynthesis-male-outfit2-mono-4gpus.conf --base_exp_dir exp/SelfreconSynthesis-male_outfit2_1711697167_slurm_mono_1_1_3_true

# # female-4
# python extract_geometry.py --device cuda:3 --conf confs/hfavatar-people_snapshot/PeopleSnapshot-female-4-casual-mono-4gpus.conf --base_exp_dir exp/Peoplesnapshot-female-4-casual_1710657384_slurm_mono_1_1_3_true > female-4.log 2>&1

# # female-3
# python extract_geometry.py --device cuda:1 --conf confs/hfavatar-people_snapshot/PeopleSnapshot-female-3-casual-mono-4gpus.conf --base_exp_dir exp/Peoplesnapshot-female-3-casual_1711463792_slurm_mono_1_1_3_true > female-3.log 2>&1
