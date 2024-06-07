from typing import Optional
from tap import Tap
import os
import random
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from Trainer.vit import MaskGIT
from image_utils import library_ops
from decoupled_utils import breakpoint_on_error
from pathlib import Path
import time

class ArgumentParser(Tap):
    data: str = "cifar10"
    data_folder: str = ""
    vqgan_folder: str = ""
    vit_folder: str = ""
    output_dir: Path = Path("logs")
    sched_mode: str = "arccos"
    grad_cum: int = 1
    channel: int = 3
    num_workers: int = 8
    step: int = 8
    seed: int = 42
    epoch: int = 300
    img_size: int = 256
    bsize: int = 256
    mask_value: int = 1024
    lr: float = 1e-4
    cfg_w: float = 3
    r_temp: float = 4.5
    sm_temp: float = 1.0
    drop_label: float = 0.1
    test_only: bool = False
    resume: bool = False
    debug: bool = False
    dtype: str = "bfloat16"
    log_iter: int = 2500
    overfit: bool = False
    wandb: bool = False
    unified_model: bool = False
    text_seq_len: int = 40
    text_vocab_size: int = 41
    run_name: Optional[str] = None
    ckpt_path: Optional[str] = None

def main(args):
    """ Main function: Train or eval MaskGIT """
    maskgit = MaskGIT(args)

    if args.test_only:  # Evaluate the networks
        maskgit.eval()
    elif args.debug:  # custom code for testing inference
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 635, random.randint(0, 999)] * 1, "r_row"
            labels = torch.LongTensor(labels).to(args.device)
            sm_temp = 1.3          # Softmax Temperature
            r_temp = 7             # Gumbel Temperature
            w = 9                  # Classifier Free Guidance
            randomize = "linear"   # Noise scheduler
            step = 32              # Number of step
            sched_mode = "arccos"  # Mode of the scheduler
            # Generate sample
            gen_sample, _, _ = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, r_temp=r_temp, w=w,
                                              randomize=randomize, sched_mode=sched_mode, step=step)
            gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
            # Save image
            save_image(gen_sample, f"saved_img/sched_{sched_mode}_step={step}_temp={sm_temp}"
                                   f"_w={w}_randomize={randomize}_{name}.jpg")
    else:  # Begin training
        maskgit.fit()

def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()

if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0
    args.run_name = f"{args.run_name}_" if args.run_name else ""
    args.run_name = f"{args.run_name}{args.data}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = Path('logs') / f"{args.run_name}"

    if args.seed > 0:  # Set the seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    world_size = torch.cuda.device_count()

    with breakpoint_on_error():
        if world_size > 1:  # launch multi training
            print(f"{world_size} GPU(s) found, launch multi-gpus training")
            args.is_multi_gpus = True
            launch_multi_main(args)
        else:  # launch single Gpu training
            print(f"{world_size} GPU found")
            args.is_master = True
            args.is_multi_gpus = False
            main(args)
