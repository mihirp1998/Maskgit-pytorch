import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA


class CelebADataset(Dataset):
    def __init__(
        self,
        root="/scratch/aswerdlo/data",
        overfit=True,
        split="train",
        **kwargs
    ):
        self.dataset = CelebA(root=root, split=split, target_type=["attr", "identity", "bbox", "landmarks"], download=False, **kwargs)
        self.identity_dict = {i: str(i) for i in range(100000)}
        self.use_custom_tokenizer = True
        self.overfit = overfit
        self.pad_token = 40
        self.attribute_strings = [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ]

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
        tokens = torch.full_like(attr, dtype=torch.int64, fill_value=self.pad_token)  # 40 is our pad token
        true_attributes = attr.nonzero().squeeze(-1)
        tokens[: len(true_attributes)] = true_attributes
        return tokens

    def decode(self, tokens):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        decoded_text = []
        for i in range(tokens.shape[0]):
            tokens_ = tokens[i].detach().cpu().tolist()
            decoded_text.append(", ".join(f"{self.attribute_strings[i]}" for i in tokens_ if i < self.pad_token))

        return decoded_text[0] if tokens.ndim == 1 else decoded_text
