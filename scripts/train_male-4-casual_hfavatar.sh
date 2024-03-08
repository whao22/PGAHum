user_comment='mono_1_1_3_true'
exp_comment=`date +%s`
run_name="CoreView_male-4-casual_${exp_comment}_run_${user_comment}"
base_exp_dir="exp/${run_name}"
conf="confs/hfavatar-people_snapshot/PeopleSnapshot-male-4-casual-mono-4gpus.conf"
log_name="CoreView_male-4-casual_${exp_comment}_run.log"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf ${conf}  \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name} > ${log_name} 2>&1