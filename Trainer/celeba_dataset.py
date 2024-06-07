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
        root="/home/aswerdlo/repos/lib/UniD3/data",
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
    
    def idxs_to_attributes(self, idxs, mask_token_idx: int = 1024):
        if any(idx > self.pad_token and idx != mask_token_idx for idx in idxs):
            assert False, f"Found invalid index: {idxs}"
        return ", ".join([self.attribute_strings[idx] if idx < self.pad_token else ("Mask" if idx == mask_token_idx else "Pad") for idx in idxs])

    def decode(self, tokens, mask_token_idx: int = 1024):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        decoded_text = []
        for i in range(tokens.shape[0]):
            tokens_ = tokens[i].detach().cpu().tolist()
            decoded_str = ", ".join("Mask" if i == mask_token_idx else f"{self.attribute_strings[i]}" for i in tokens_ if i < self.pad_token or i == mask_token_idx)
            if decoded_str == "":
                decoded_str = "No attributes"
            decoded_text.append(decoded_str)
        return decoded_text[0] if tokens.ndim == 1 else decoded_text
