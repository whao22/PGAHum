cd /home/wanghao/workspace/HF-Avatar

user_comment='mono_1_1_3_true'
exp_comment=`date +%s`
run_name="SyntheticHuman-megan_1710946590_slurm_${user_comment}"
base_exp_dir="exp/${run_name}"
conf="confs/hfavatar-synthetic_human/SyntheticHuman-megan-mono-4gpus.conf"
log_name="SyntheticHuman-megan_${exp_comment}_run.log"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf ${conf} \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name} > ${log_name} 2>&1