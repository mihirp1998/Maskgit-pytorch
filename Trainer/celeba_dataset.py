import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA


class CelebADataset(Dataset):
    def __init__(self, root='/scratch/aswerdlo/data', overfit=True, split='train', size=256, tokenizer=None, downstream='diffusion+vqvae', target_type=['attr', 'identity', 'bbox', 'landmarks'], **kwargs):
        self.dataset = CelebA(root=root, split=split, target_type=target_type, download=False, **kwargs)
        self.attr_dict = {i: str(i) for i in range(40)}
        self.identity_dict = {i: str(i) for i in range(100000)}
        self.use_custom_tokenizer = True
        self.tokenizer = tokenizer

        if self.tokenizer is None and self.use_custom_tokenizer:
            from Trainer.tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(bpe_path='/home/aswerdlo/repos/lib/UniD3/misc/bpe_simple_vocab_16e6.txt.gz', token_length=7936)

        self.downstream = downstream
        self.overfit = overfit
        self.pad_token = 40
        self.attribute_strings = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, index):
        image, target = self.dataset[0 if self.overfit else index]

        if self.use_custom_tokenizer:
            tokenized_text = self.custom_tokenizer(target)
        else:
            caption = self.generate_caption(target)
            tokenized_text = self.tokenizer.tokenize(caption, 128).squeeze(0)

        return (image, tokenized_text)
    
    def custom_tokenizer(self, target):
        attr, identity, bbox, landmarks = target
        assert attr.shape[0] == 40
        tokens = torch.full_like(attr, dtype=torch.int32, fill_value=self.pad_token) # 40 is our pad token
        true_attributes = attr.nonzero().squeeze(-1)
        tokens[:len(true_attributes)] = true_attributes
        return tokens
    
    def decode_custom_tokenizer(self, tokens):
        tokens = tokens.detach().cpu().tolist()
        return ", ".join(f"Attribute {self.attr_dict[i]} is true" for i in tokens if i < self.pad_token)

    def generate_caption(self, target):
        attr, identity, bbox, landmarks = target

        # Generate attribute part of the caption
        attr_caption = [f"has {self.attr_dict[i]}" for i, a in enumerate(attr) if a == 1]
        attr_caption = ', '.join(attr_caption)

        # Generate identity part of the caption
        identity_caption = self.identity_dict[identity.item()]

        # Generate bounding box part of the caption
        bbox_caption = f"Bounding Box: (x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]})"

        # Generate landmarks part of the caption
        landmark_caption = [f"{name}: ({landmarks[i*2]}, {landmarks[i*2+1]})" for i, name in enumerate(['lefteye', 'righteye', 'nose', 'leftmouth', 'rightmouth'])]
        landmark_caption = ', '.join(landmark_caption)

        # Combine all parts to form the final caption
        caption = f"{identity_caption}, {attr_caption}, {bbox_caption}, {landmark_caption}"

        return caption[:127] # just to be safe since they use 128 text tokens
