user_comment='mono_winner_wdeltasdf'
exp_comment=`date +%s`
run_name="CoreView_392_${exp_comment}_run_${user_comment}"
base_exp_dir="exp/${run_name}"
log_name="CoreView_392_${exp_comment}_run.log"
conf="confs/hfavatar-zju/ZJUMOCAP-392-4gpus.conf"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf ${conf}  \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name} > ${log_name} 2>&1