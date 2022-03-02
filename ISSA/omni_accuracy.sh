for c_dim in 1000; do
for nz in 1000; do
for sample_size in 10; do
for num_train in 18; do
for num_gen in 100; do
for num_sets in 10; do
for anno_rate in 1; do
for aug in  'translation,cutout' ; do
for model_dir in "specify where to load the model from"; do
for seed in 2020; do
for n_epochs in 1000; do
for c_scale in 0.1; do
for g_scale in 1; do
for sn in 1; do
for ngf in 400; do
for ndf in 400; do

cmd="omni_accuracy.py \
--seed $seed \
--c_dim $c_dim \
--nz $nz \
--num_gen $num_gen \
--num_sets $num_sets \
--sample_size $sample_size \
--num_train $num_train \
--anno_rate $anno_rate \
--augments $aug \
--model_dir $model_dir \
--n_epochs $n_epochs \
--c_scale $c_scale \
--g_z_scale $g_scale \
--g_sn $sn \
--ngf $ngf \
--ndf $ndf \
"
if [ "$1" == 1 ]
then
    echo $cmd
    CUDA_VISIBLE_DEVICES=0 python -u $cmd |& tee $2.txt
else    
    echo "write your own script here"
fi
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done