user_comment='w/_inner-w/_init_sdf-rgb10-lpip0.01-eik0.01'
exp_comment=`date +%s`
run_name="CoreView_377_${exp_comment}_run_${user_comment}"
base_exp_dir="exp/${run_name}"
log_name="CoreView_377_${exp_comment}_run.log"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf confs/arah-zju/ZJUMOCAP-377-mono-4gpus.conf  \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name} > ${log_name} 2>&1