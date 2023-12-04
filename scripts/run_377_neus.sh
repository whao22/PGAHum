exp_comment=`date +%s`
run_name="CoreView_377_${exp_comment}_run"
base_exp_dir="exp/${run_name}"
echo The base experiment directory is ${base_exp_dir}.

python train.py \
    --conf confs/arah-zju/ZJUMOCAP-377-mono-4gpus.conf  \
    --base_exp_dir ${base_exp_dir} \
    --run_name ${run_name}