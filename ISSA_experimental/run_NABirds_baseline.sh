out_dir="experiment_name"

cmd="train.py \
--outdir=${out_dir} \
--data=./celeba_train.zip \
--gpus=1 \
--encoder=1 \
--cond=0 \
--ssize=1 \
--disable_pl_mixing=1 \
--cond_identity_layers=0 \
--loss_fcn=loss_orig \
--encoder_name=ISSAEncoder \
--channel_base=16384 \
--channel_max=256 \
--e_batchnorm=1 \
--g_batchnorm=0 \
--d_batchnorm=0 \
--classifier_init=0 \
--z_weight=0.0 \
--aug=noaug \
--n_map_layers=2 \
--ge_lr=2e-4 \
--d_lr=2e-4"


python $cmd