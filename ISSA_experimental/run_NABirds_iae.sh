out_dir="experiment_name"

cmd="train_issa_swap.py \
--outdir=${out_dir} \
--data=./celeba_train_crop.zip \
--gpus=1 \
--encoder=1 \
--cond=0 \
--ssize=5 \
--batch=60 \
--disable_pl_mixing=1 \
--cond_identity_layers=0 \
--loss_fcn=loss_issa \
--generator_name=Generator \
--encoder_name=Encoder \
--disc_name=Discriminator \
--channel_base=16384 \
--channel_max=256 \
--z_weight=1.0 \
--aug=noaug \
--n_map_layers=2 \
--ge_lr=0.0025 \
--d_lr=0.0025 \
--nodiscpac=1"


python $cmd