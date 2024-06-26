# Trainer for MaskGIT
import os
import random
import time
import math


import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

from Trainer.trainer import Trainer
from Network.transformer import MaskTransformer

from Network.Taming.models.vqgan import VQModel


class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args                                                        # Main argument see main.py
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        self.ae = self.get_network("autoencoder")
        self.codebook_size = self.ae.n_embed   
        print("Acquired codebook size:", self.codebook_size)   
        self.vit = self.get_network("vit")                                      # Load Masked Bidirectional Transformer   
        self.patch_size = self.args.img_size // 2**(self.ae.encoder.num_resolutions-1)     # Load VQGAN
        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)    # Get cross entropy loss
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay

        self.text_seq_len = self.args.text_seq_len
        self.text_vocab_size = self.args.text_vocab_size
        
        # Load data if aim to train or test the model
        if not self.args.debug:
            self.train_data, self.test_data = self.get_data()

        # Initialize evaluation object if testing
        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            self.sae = SampleAndEval(device=self.args.device, num_images=50_000)

        self.dtype = torch.bfloat16 if self.args.dtype == "bfloat16" else (torch.float16 if self.args.dtype == "float16" else torch.float32)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            model = MaskTransformer(
                # Small
                img_size=self.args.img_size, hidden_dim=768, codebook_size=self.codebook_size, depth=24, heads=16, mlp_dim=3072, dropout=0.1, text_tokens=self.args.text_vocab_size, text_seqlen=self.args.text_seq_len
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
            )

            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
                if self.args.is_master:
                    print("load ckpt and resume from:", ckpt)
                # Read checkpoint file
                checkpoint = torch.load(ckpt, map_location='cpu')
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                # Load network
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif self.args.ckpt_path:
                if self.args.is_master:
                    print("load ckpt from:", self.args.ckpt_path)
                ckpt = torch.load(self.args.ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'], strict=False)

            model = model.to(self.args.device)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "autoencoder":
            # Load config
            config = OmegaConf.load(self.args.vqgan_folder + "model.yaml")
            model = VQModel(**config.model.params)
            checkpoint = torch.load(self.args.vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
            # Load network
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)
            

            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=256):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        return mask_code, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=2500):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss = 0.
        # st()
        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        for x, y in bar:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            assert x.min() >= (0 - torch.finfo(self.dtype).eps) and x.max() <= (1 + torch.finfo(self.dtype).eps)
            x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            # Drop xx% of the condition for cfg
            drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

            # VQGAN encoding to img tokens
            with torch.no_grad():
                emb, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask the encoded tokens
            masked_code, mask = self.get_mask_code(code, value=self.args.mask_value, codebook_size=self.codebook_size)

            masked_text_code = None
            if self.args.unified_model:
                text_code = y
                assert text_code.min() >= 0 and text_code.max() < self.text_vocab_size
                masked_text_code, text_mask = self.get_mask_code(text_code.unsqueeze(-1), value=self.args.mask_value, codebook_size=self.text_vocab_size)
                masked_text_code, text_mask = masked_text_code.squeeze(-1), text_mask.squeeze(-1)
                # print(0, masked_text_code[0].tolist())

            with torch.cuda.amp.autocast(dtype=self.dtype):                         # half precision                
                pred, pred_text = self.vit(masked_code, y, text_code=masked_text_code, drop_label=drop_label)  # The unmasked tokens prediction
                
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1)) / self.args.grad_cum

                if self.args.unified_model:
                    text_loss = self.criterion(pred_text.reshape(-1, self.text_vocab_size), text_code.view(-1)) / self.args.grad_cum
                    loss = (loss + text_loss)/2

            # update weight if accumulation of gradient is done
            update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)                      # rescale loss
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            # logs
            if update_grad and self.args.is_master:
                self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

            if self.args.wandb and self.args.iter and self.args.is_master:
                import wandb
                wandb.log({"loss": loss}, step=self.args.iter)
            try:
                if self.args.iter % log_iter == 0 and self.args.is_master:
                    # Generate sample for visualization
                    gen_sample, l_codes, l_mask, gen_text, t_codes, t_mask = self.sample(nb_sample=10)
                    gen_sample_grid = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/Sampling", gen_sample_grid, self.args.iter)
                    # Show reconstruction
                    unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                    reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
                    reco_sample_grid = vutils.make_grid(reco_sample.data, nrow=10, padding=2, normalize=True)
                    self.log_add_img("Images/Reconstruction", reco_sample_grid, self.args.iter)

                    if self.args.wandb:
                        import wandb
                        code_imgs, masked_code_imgs, unmasked_code_imgs = torch.split(reco_sample, 10)
                    
                        gt_captions = self.train_data.dataset.decode(text_code)
                        masked_captions = self.train_data.dataset.decode(masked_text_code)
                        pred_text_code = torch.softmax(pred_text, -1).max(-1)[1]
                        unmasked_captions = self.train_data.dataset.decode(pred_text_code)
                        
                        def get_attributes(_code):
                            _attributes = set(torch.unique(_code).tolist())
                            if self.text_vocab_size in _attributes:
                                _attributes.remove(self.text_vocab_size)
                            if self.args.mask_value in _attributes:
                                _attributes.remove(self.args.mask_value)
                            return _attributes

                        avg_precision = 0
                        avg_recall = 0
                        correct_attributes = []
                        correct_new_attributes = []
                        missing_attributes = []
                        wrong_attributes = []
                        recalls = []
                        precisions = []
                        for i in range(text_code.shape[0]):
                            gt_attributes = get_attributes(text_code[i])
                            input_attributes = get_attributes(masked_text_code[i])
                            pred_attributes = get_attributes(pred_text_code[i])

                            correct_attributes.append(self.train_data.dataset.idxs_to_attributes(sorted(list())))
                            correct_new_attributes.append(self.train_data.dataset.idxs_to_attributes(sorted(list(
                                (gt_attributes & pred_attributes) - input_attributes
                            ))))
                            missing_attributes.append(self.train_data.dataset.idxs_to_attributes(sorted(list(gt_attributes - pred_attributes))))
                            wrong_attributes.append(self.train_data.dataset.idxs_to_attributes(sorted(list(pred_attributes - gt_attributes))))

                            precision = len(gt_attributes & pred_attributes) / len(pred_attributes) if len(pred_attributes) > 0 else 0
                            recall = len(gt_attributes & pred_attributes) / len(gt_attributes) if len(gt_attributes) > 0 else 0

                            precisions.append(precision)
                            recalls.append(recall)

                        avg_precision = sum(precisions) / len(precisions)
                        avg_recall = sum(recalls) / len(recalls)

                        mask_values = torch.sum(~mask, dim=(1, 2)) / mask[0].numel()

                        pred_table = wandb.Table(
                            columns=[
                                "GT Reconstruction Image", 
                                "GT Reconstruction Caption", 
                                "Masked [Input] Image", 
                                "Masked [Input] Caption", 
                                "Unmasked [Pred] Image", 
                                "Unmasked [Pred] Caption", 
                                "Correct New Attributes", 
                                "Correct Attributes", 
                                "Missing Attributes", 
                                "Wrong Attributes", 
                                "Mask Value", 
                                "Precision", 
                                "Recall"
                            ]
                        )
                        for gt_img, gt_caption, masked_img, masked_caption, img, caption, correct_new_attribute, correct, missing, wrong, mask_value, precision, recall in zip(code_imgs, gt_captions, masked_code_imgs, masked_captions, unmasked_code_imgs, unmasked_captions, correct_new_attributes, correct_attributes, missing_attributes, wrong_attributes, mask_values, precisions, recalls):
                            pred_table.add_data(wandb.Image(gt_img), gt_caption, wandb.Image(masked_img), masked_caption, wandb.Image(img), caption, correct_new_attribute, correct, missing, wrong, mask_value, precision, recall)

                        gen_table = wandb.Table(columns=["Uncond Sampled Image", "Uncond Sampled Caption"])
                        gen_captions = self.train_data.dataset.decode(gen_text)
                        for img, caption in zip(gen_sample, gen_captions):
                            gen_table.add_data(wandb.Image(img), caption)

                        wandb.log({
                            "uncond_sample_table": gen_table,
                            "pred_table": pred_table,
                            "mean_precision": avg_precision,
                            "mean_recall": avg_recall
                        }, step=self.args.iter)

                    # Save Network
                    if self.args.iter > 0:
                        def save_ckpt(_path):
                            self.save_network(
                                model=self.vit,
                                path=_path,
                                iter=self.args.iter,
                                optimizer=self.optim,
                                global_epoch=self.args.global_epoch
                            )

                        save_ckpt(self.args.output_dir / "current.pth")
                        if self.args.global_epoch % 20 == 0:
                            save_ckpt(self.args.output_dir / f"epoch_{self.args.global_epoch:03d}.pth")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)

            self.args.iter += 1

        return cum_loss / n

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            if self.args.wandb:
                import wandb
                wandb.init(config=self.args, project="maskgit", name=self.args.run_name)
                wandb.watch(self.vit, log_freq=200)
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            train_loss = self.train_one_epoch(log_iter=self.args.log_iter)

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            # Save model
            if e % 10 == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', train_loss, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Loss {train_loss:.4f},"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1

    def eval(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        # Evaluate the model
        m = self.sae.compute_and_log_metrics(self)
        self.vit.train()
        return m

    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                # Decoding masked code
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0,  self.codebook_size-1))
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        text_l_codes = []
        text_l_mask = []
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                assert False
                code = init_code
                mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                    text_code = torch.full((nb_sample, self.text_seq_len), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size + self.text_seq_len ).to(self.args.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode
            # st()
            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast(dtype=self.dtype):  # half precision
                    if w != 0:
                        # Model Prediction
                        logit, text_logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([text_code, text_code], dim=0),
                                         torch.cat([~drop, drop], dim=0))

                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                        text_logit_c, text_logit_u = torch.chunk(text_logit, 2, dim=0)
                        # Classifier Free Guidance
                        text_logit = (1 + _w) * text_logit_c - _w * text_logit_u
                    else:
                        logit, text_logit = self.vit(code.clone(), labels, text_code, drop_label=~drop)
                
                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()
                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))
                
                text_prob = torch.softmax(text_logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                text_distri = torch.distributions.Categorical(probs=text_prob)
                text_code = text_distri.sample()                
                text_conf = torch.gather(text_prob, 2, text_code.view(nb_sample, self.text_seq_len, 1))

                merged_pred_code = torch.cat([pred_code,text_code],dim=1)
                merged_conf = torch.cat([conf, text_conf], dim=1)
                # st()

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size + self.text_seq_len)) * (1 - ratio)
                    merged_conf = torch.log(merged_conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    merged_conf = torch.rand_like(merged_conf) if indice < 2 else merged_conf
                elif randomize == "random":   # chose random prediction at each step
                    merged_conf = torch.rand_like(merged_conf)
                # st()
                # do not predict on already predicted tokens
                merged_conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                image_conf = merged_conf[:,:self.patch_size* self.patch_size]
                image_mask = mask[:,:self.patch_size* self.patch_size]
                text_conf = merged_conf[:,self.patch_size* self.patch_size:]
                text_mask = mask[:,self.patch_size* self.patch_size:]
                
                image_conf = (image_conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_image_mask = (image_mask.view(nb_sample, self.patch_size, self.patch_size).float() * image_conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_image_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_image_mask]
                
                text_conf = (text_conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.text_seq_len)
                f_text_mask = (text_mask.view(nb_sample, self.text_seq_len).float() * text_conf.view(nb_sample, self.text_seq_len).float()).bool()
                text_code[f_text_mask] = text_code.view(nb_sample, self.text_seq_len)[f_text_mask]                

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask[:,:self.patch_size* self.patch_size].view(nb_sample, self.patch_size, self.patch_size).clone())
                
                text_l_codes.append(text_code.view(nb_sample, self.text_seq_len).clone())
                text_l_mask.append(mask[:,self.patch_size* self.patch_size:].view(nb_sample, self.text_seq_len).clone())                

            # decode the final prediction
            _code = torch.clamp(code, 0,  self.codebook_size-1)
            x = self.ae.decode_code(_code)

        self.vit.train()
        return x, l_codes, l_mask, text_code, text_l_codes, text_l_mask
