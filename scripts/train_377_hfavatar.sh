user_comment='mono_v007_tri-scale0.1_nn_intermask'
exp_comment=`date +%s`
run_name="CoreView_377_${exp_comment}_run_${user_comment}"
base_exp_dir="exp/${run_name}"
conf="confs/hfavatar-zju/ZJUMOCAP-377-mono-4gpus.conf"
log_name="CoreView_377_${exp_comment}_run.log"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf ${conf}  \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name} > ${log_name} 2>&1