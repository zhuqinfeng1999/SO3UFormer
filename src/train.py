from __future__ import absolute_import, division, print_function

import math
import argparse
import os
import torch.distributed as dist


parser = argparse.ArgumentParser(description="360 Degree Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")

# data settings
parser.add_argument("--task", type=str, default="depth", choices=["depth", "segmentation"])
parser.add_argument("--dataset_name", type=str, default="stanford2d3d")
parser.add_argument("--dataset_root_dir", type=str, help="root location for the data")

# model settings
parser.add_argument("--mode", type=str, default="vertex", choices=["face", "vertex"], help="folder to save the model in")
parser.add_argument("--img_rank", type=int, default=7)
parser.add_argument("--img_width", type=int, default=512)
parser.add_argument("--num_scales", type=int, default=4)
parser.add_argument("--win_size_coef", type=int, default=2)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--abs_pos_enc_in", type=int, default=True)
parser.add_argument("--abs_pos_enc", type=int, default=True)
parser.add_argument("--rel_pos_bias", type=int, default=True)
parser.add_argument("--rel_pos_bias_size", type=int, default=7)
parser.add_argument("--rel_pos_init_variance", type=float, default=1)
parser.add_argument("--d_head_coef", type=int, default=2)
parser.add_argument("--enc_num_heads", nargs="+", type=int, default=[2,4,8,16])
parser.add_argument("--dec_num_heads", nargs="+", type=int, default=[16,16,8,4])
parser.add_argument("--bottleneck_num_heads", type=int, default=None)
parser.add_argument("--scale_depth", type=int, default=2)
parser.add_argument("--debug_skip_attn", type=int, default=False)
parser.add_argument("--append_self", type=int, default=False)
parser.add_argument("--use_checkpoint", type=int, default=True)

parser.add_argument("--dr", type=float, default=0.)
parser.add_argument("--dpr", type=float, default=0.)
parser.add_argument("--adr", type=float, default=0.)
parser.add_argument("--aodr", type=float, default=0.)
parser.add_argument("--posdr", type=float, default=0.)

# parser.add_argument("--abs_pos_enc_in", nargs="+", type=str, default=None)
# parser.add_argument("--abs_pos_enc", nargs="+", type=str, default=None)

parser.add_argument("--downsample", type=str, default="center")
parser.add_argument("--upsample", type=str, default="interpolate")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--ltr", dest="limit_train_batches", type=int, default=math.inf, help="limit train batches")
parser.add_argument("--train_batch_size", type=int, default=16, help="batch size")#ori 16
parser.add_argument("--val_batch_size", type=int, default=10, help="batch size")
parser.add_argument("--num_epochs", type=int, default=400, help="number of epochs")
parser.add_argument("--accum_grads", type=int, default=1, help="number of epochs")


# loading and logging settings
parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
parser.add_argument("--enable_save", type=int, default=True, help="save model")
parser.add_argument("--save_frequency", type=int, default=100, help="number of epochs between each save")
parser.add_argument("--load_weights_task", type=str, default=None)
parser.add_argument("--load_weights_path", type=str, default=None, help="path to local model weights (.pth)")

# ablation/scaffold flags (no behavior change yet)
parser.add_argument("--use_quadrature_attn", type=int, default=0, help="enable quadrature attention (scaffold)")
parser.add_argument("--quadrature_mode", type=str, default="logit", choices=["logit", "value", "value_renorm"], help="quadrature attention mode")
parser.add_argument("--use_abs_phi_pe", type=int, default=1, help="enable absolute phi positional encoding (scaffold)")
parser.add_argument("--rel_pos_bias_type", type=str, default="grid7_dtheta_dphi", help="relative bias type (scaffold)")
parser.add_argument("--rel_pos_bins", type=int, default=32, help="relative bias bins (scaffold)")
parser.add_argument("--gauge_num_frames", type=int, default=3, help="gauge frame count (scaffold)")
parser.add_argument("--gauge_m_max", type=int, default=2, help="gauge max order (scaffold)")
parser.add_argument("--gauge_mode", type=str, default="pool_invariant", choices=["pool_invariant", "c6_equivariant"], help="gauge bias mode")
parser.add_argument("--gauge_anchor_mode", type=str, default="tangent_max", choices=["index", "geodesic", "tangent_max"], help="gauge anchor selection mode")
parser.add_argument("--gauge_debug", type=int, default=0, help="print gauge debug stats once")
parser.add_argument("--downsample_mode", type=str, default="current_default", help="downsample mode override (scaffold)")
parser.add_argument("--upsample_mode", type=str, default="current_default", help="upsample mode override (scaffold)")
parser.add_argument("--upsample_sigma", type=float, default=0.4, help="geodesic upsample kernel sigma (radians)")
parser.add_argument("--eq_loss_weight", type=float, default=0.0, help="equivariance loss weight (scaffold)")
parser.add_argument("--eq_loss_samples", type=int, default=1, help="equivariance loss samples (scaffold)")

# data augmentation settings
parser.add_argument("--disable_color_augmentation",  dest="color_augmentation", action="store_false",
                    help="if set, do not use color augmentation")
parser.add_argument("--disable_lr_flip_augmentation", dest="lr_flip_augmentation", action="store_false",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", dest="yaw_rotation_augmentation", action="store_false",
                    help="if set, do not use yaw rotation augmentation")

# wandb settings
parser.add_argument("--exp_name", default="train_sphereuformer", type=str)
parser.add_argument("--log_dir", type=str, help="log directory")
parser.add_argument("--wandb_entity", type=str)
parser.add_argument("--wandb_project", type=str)
parser.add_argument("--wandb_group", default=None, type=str)


parser.add_argument("--vis_color_map", type=str, default="viridis", help="color map for depth visualization")
parser.add_argument("--vis_color_map_invert", action="store_true", help="invert color map for depth visualization")


parser.add_argument("--no_gpu", dest="use_gpu", action="store_false")
parser.add_argument("--test", dest="test", action="store_true")
parser.add_argument("--distributed", action="store_true", help="enable DistributedDataParallel (torchrun)")
parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"], help="which split to evaluate on (val/test)")

args = parser.parse_args()


def _normalize_quadrature_mode(mode: str) -> str:
    if mode == "value":
        return "value_renorm"
    return mode


def _print_final_config(args):
    print("Final ablation config:")
    print(f"  use_quadrature_attn={bool(args.use_quadrature_attn)}")
    print(f"  quadrature_mode={args.quadrature_mode}")
    print(f"  use_abs_phi_pe={bool(args.use_abs_phi_pe)}")
    print(f"  rel_pos_bias_type={args.rel_pos_bias_type}")
    print(f"  rel_pos_bins={args.rel_pos_bins}")
    print(f"  gauge_mode={args.gauge_mode}")
    print(f"  gauge_num_frames={args.gauge_num_frames}")
    print(f"  gauge_m_max={args.gauge_m_max}")
    print(f"  gauge_anchor_mode={args.gauge_anchor_mode}")
    print(f"  gauge_debug={bool(args.gauge_debug)}")
    print(f"  downsample_mode={args.downsample_mode}")
    print(f"  upsample_mode={args.upsample_mode}")
    print(f"  upsample_sigma={args.upsample_sigma}")
    print(f"  eq_loss_weight={args.eq_loss_weight}")
    print(f"  eq_loss_samples={args.eq_loss_samples}")


def main():
    # Detect distributed launch via torchrun env vars
    args.distributed = args.distributed or ("LOCAL_RANK" in os.environ)
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.distributed:
        assert args.use_gpu, "Distributed training requires CUDA"
        torch_device_idx = args.local_rank
        import torch
        torch.cuda.set_device(torch_device_idx)
        dist.init_process_group(backend="nccl")

    args.quadrature_mode = _normalize_quadrature_mode(args.quadrature_mode)
    if not args.distributed or args.rank == 0:
        # Keep this print in sync with suite flags and model builder plumbing.
        _print_final_config(args)

    if args.task == "depth":
        from trainer_dep import Trainer
    elif args.task == "segmentation":
        from trainer_seg import Trainer
    else:
        raise NotImplementedError(args.task)

    trainer = Trainer(args)

    if not args.test:
        trainer.train()
    else:
        trainer.test()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
