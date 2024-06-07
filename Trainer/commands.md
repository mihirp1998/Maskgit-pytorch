

python main.py --data=celeba --vqgan-folder="pretrained_maskgit/VQGAN/" --vit-folder="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth" --bsize=32 --drop-label=1.0 --cfg_w=0 --writer-log=logs/v0 --log_iter=1 --overfit --wandb --unified_model --text_seq_len=40 --text_vocab_size=41

