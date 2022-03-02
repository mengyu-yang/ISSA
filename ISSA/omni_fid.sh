for c_dim in 1000; do
for nz in 1000; do
for sample_size in 10; do
for num_support in 10; do
for num_gen in 100; do
for num_sets in 10; do
for seed in 2020; do
for model_dir in "specify where the model is stored"; do
for c_scale in 0.1; do
for g_scale in 1; do
for sn in 1; do
for ngf in 400; do
for ndf in 400; do
method="ISSA"
d="fid"
SLURM_TMPDIR=""
output_dir=${SLURM_TMPDIR}/nngan/${d}/${method}/$2

cmd="omni_fid.py \
--output_dir ${output_dir} \
--num_support $num_support \
--c_dim $c_dim \
--nz $nz \
--sample_size $sample_size \
--num_sets $num_sets \
--num_gen $num_gen \
--seed $seed \
--model_dir $model_dir \
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
    echo "write your own script"
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