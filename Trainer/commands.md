

python main.py --data=celeba --vqgan_folder="pretrained_maskgit/VQGAN/" --ckpt_path="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pthcurrent.pth" --bsize=32 --drop_label=1.0 --cfg_w=0 --log_iter=100 --wandb --unified_model --text_seq_len=40 --text_vocab_size=41




torchrun --standalone --nnodes=1 --nproc_per_node=7 main.py --data=celeba --vqgan_folder="pretrained_maskgit/VQGAN/" --vit_folder="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth" --bsize=64 --drop_label=1.0 --cfg_w=0 --log_iter=100 --wandb --unified_model --text_seq_len=40 --text_vocab_size=41 --run_name="run_v0"