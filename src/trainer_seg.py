import os
import time
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import wandb

from data.get_dataloaders import get_dataloaders
from metrics.segmentation import Evaluator
from network.sphere_model import SO3UFormer
from torch.utils.data.distributed import DistributedSampler


class Trainer:
    def __init__(self, args):
        self.args = args

        self.distributed = getattr(args, "distributed", False)
        self.rank = getattr(args, "rank", 0)
        self.world_size = getattr(args, "world_size", 1)
        self.local_rank = getattr(args, "local_rank", 0)

        if args.use_gpu:
            if self.distributed:
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.config = dict(
            img_rank=self.args.img_rank,
            img_width=self.args.img_width,
            node_type=self.args.mode,
            num_scales=self.args.num_scales,
            win_size_coef=self.args.win_size_coef,
            scale_factor=self.args.scale_factor,
            downsample=self.args.downsample,
            scale_depth=self.args.scale_depth,
        )
        # Keep these in sync with train.py flags and suite scripts for reproducible ablations.
        self.ablation_config = dict(
            use_quadrature_attn=bool(self.args.use_quadrature_attn),
            quadrature_mode=self.args.quadrature_mode,
            use_abs_phi_pe=bool(self.args.use_abs_phi_pe),
            rel_pos_bias_type=self.args.rel_pos_bias_type,
            rel_pos_bins=self.args.rel_pos_bins,
            gauge_mode=self.args.gauge_mode,
            gauge_num_frames=self.args.gauge_num_frames,
            gauge_m_max=self.args.gauge_m_max,
            gauge_anchor_mode=self.args.gauge_anchor_mode,
            gauge_debug=bool(self.args.gauge_debug),
            downsample_mode=self.args.downsample_mode,
            upsample_mode=self.args.upsample_mode,
            upsample_sigma=self.args.upsample_sigma,
            eq_loss_weight=self.args.eq_loss_weight,
            eq_loss_samples=self.args.eq_loss_samples,
        )

        sphere_img_rank = self.args.img_rank
        grid_img_width = self.args.img_width

        self.wandb_run = None
        os.makedirs(args.log_dir, exist_ok=True)
        if args.wandb_project and (not self.distributed or self.rank == 0):
            self.wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.exp_name,
                group=args.wandb_group,
                dir=args.log_dir,
            )
            self._run = wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{self.wandb_run.id}")
        # 本地日志文件（只在 rank0 写）
        self.log_file = os.path.join(args.log_dir, "training_log.txt") if (not self.distributed or self.rank == 0) else None
        if self.log_file and not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("mode,step,epoch,metrics\n")

        # Configure data
        self.loader_train, self.loader_val = get_dataloaders(
            dataset_name=self.args.dataset_name,
            dataset_root_dir=self.args.dataset_root_dir,
            dataset_kwargs={
                "sphere_rank": sphere_img_rank,
                "grid_width": grid_img_width,
                "sphere_node_type": self.config["node_type"],
            },
            augmentation_kwargs=dict(
                color_augmentation=False,
                lr_flip_augmentation=self.args.lr_flip_augmentation,
                yaw_rotation_augmentation=self.args.yaw_rotation_augmentation,
            ),
            train_batch_size=self.args.train_batch_size,
            val_batch_size=self.args.val_batch_size or self.args.train_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            train_sampler=None,
            val_sampler=None,
            shuffle_train=not self.distributed,
            shuffle_val=False,
            distributed=self.distributed,
            world_size=self.world_size,
            rank=self.rank,
            eval_split=self.args.eval_split,
        )

        self.model = SO3UFormer(
            img_rank=sphere_img_rank,
            node_type=self.config["node_type"],
            in_channels=3,
            out_channels=self.loader_train.dataset.NUM_CLASSES,
            in_scale_factor=self.args.scale_factor,
            num_scales=self.args.num_scales,
            win_size_coef=self.args.win_size_coef,
            enc_depths=self.args.scale_depth,
            dec_depths=self.args.scale_depth,
            bottleneck_depth=self.args.scale_depth,
            d_head_coef=self.args.d_head_coef,
            enc_num_heads=self.args.enc_num_heads,
            bottleneck_num_heads=self.args.bottleneck_num_heads,
            dec_num_heads=self.args.dec_num_heads,
            #
            abs_pos_enc_in=self.args.abs_pos_enc_in,
            abs_pos_enc=self.args.abs_pos_enc,
            rel_pos_bias=self.args.rel_pos_bias,
            rel_pos_bias_size=self.args.rel_pos_bias_size,
            rel_pos_init_variance=self.args.rel_pos_init_variance,
            downsample=self.args.downsample,
            upsample=self.args.upsample,
            use_quadrature_attn=self.args.use_quadrature_attn,
            quadrature_mode=self.args.quadrature_mode,
            use_abs_phi_pe=self.args.use_abs_phi_pe,
            rel_pos_bias_type=self.args.rel_pos_bias_type,
            rel_pos_bins=self.args.rel_pos_bins,
            gauge_mode=self.args.gauge_mode,
            gauge_num_frames=self.args.gauge_num_frames,
            gauge_m_max=self.args.gauge_m_max,
            gauge_anchor_mode=self.args.gauge_anchor_mode,
            gauge_debug=bool(self.args.gauge_debug),
            downsample_mode=self.args.downsample_mode,
            upsample_mode=self.args.upsample_mode,
            upsample_sigma=self.args.upsample_sigma,
            #
            drop_rate=self.args.dr,
            drop_path_rate=self.args.dpr,
            attn_drop_rate=self.args.adr,
            attn_out_drop_rate=self.args.aodr,
            pos_drop_rate=self.args.posdr,
            #
            debug_skip_attn=self.args.debug_skip_attn,
            append_self=self.args.append_self,
            use_checkpoint=self.args.use_checkpoint,
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.wandb_run:
            wandb.log({f"total_params": total_params}, step=0)

        if self.distributed:
            self.model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        else:
            if self.device.type == "cuda" and torch.cuda.device_count() > 1:
                print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
                self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters_to_train, self.args.learning_rate)

        if self.args.load_weights_path:
            self.load_model_from_path(self.args.load_weights_path)
        elif self.args.load_weights_task is not None:
            self.load_model()

        print("Training is using:\n ", self.device)
        print("Total parameters:\n ", total_params)

        self.compute_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.evaluator = Evaluator(num_classes=self.loader_train.dataset.NUM_CLASSES)
        self.best_iou = float("-inf")
        self.eq_loss_weight = float(self.args.eq_loss_weight)
        self.eq_loss_samples = int(self.args.eq_loss_samples)

        self.xyz_proj = None
        self.xyz_img = None

        if self.wandb_run:
            self.wandb_run.config.update(self.config)
            self.wandb_run.config.update(self.ablation_config)

    def inputs_to_device(self, inputs):
        keys = inputs.keys()
        keys = [key for key in keys if "depth" not in key]
        return {key: inputs[key].to(self.device) for key in keys}

    def test(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.validate()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.mini_step = 0
        self.step = 0
        self.start_time = time.time()
        self.optimizer.zero_grad()

        self.validate()
        for self.epoch in range(1, self.args.num_epochs+1):
            self.train_one_epoch()
            self.validate()
            if self.args.enable_save and self.epoch % self.args.save_frequency == 0:
                self.save_model()

        if self.wandb_run:
            for a in self._run.logged_artifacts():
                if a.type in ("model", "optimizer") and "latest" not in a.aliases:
                    a.delete()

    def train_one_epoch(self):
        """Run a single epoch of training"""
        self.model.train()

        self.evaluator.reset_eval_metrics()

        if self.distributed and isinstance(self.loader_train.sampler, DistributedSampler):
            self.loader_train.sampler.set_epoch(self.epoch)

        pbar = tqdm.tqdm(self.loader_train) if (not self.distributed or self.rank == 0) else self.loader_train
        if hasattr(pbar, "set_description"):
            pbar.set_description(f"## {self.args.exp_name} ## Training Epoch_{self.epoch}")
        for batch_idx, inputs in enumerate(pbar, start=1):
            self.mini_step += 1

            inputs = self.inputs_to_device(inputs)
            outputs, losses = self.process_batch(inputs)

            # losses["loss"].backward()
            losses["loss"].div(self.args.accum_grads).backward()
            if self.mini_step % self.args.accum_grads == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step += 1

            # Track eval metrics
            with torch.no_grad():
                mask = inputs["sphere_valid_mask"]
                pred_sem = outputs["pred_sem"]
                gt_sem = inputs["sphere_gt_sem"]

                self.evaluator.compute_eval_metrics(gt_sem, pred_sem, mask, track=True)

                if self.mini_step % self.args.accum_grads == 0:
                    if self.step % self.args.log_frequency == 0:
                        errors = self.evaluator.get_results(update_best=False)
                        self.log("train", inputs, outputs, losses, errors, best_errors=None)

            if batch_idx / self.args.accum_grads >= self.args.limit_train_batches:
                break

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.loader_val) if (not self.distributed or self.rank == 0) else self.loader_val
        if hasattr(pbar, "set_description"):
            pbar.set_description(f"Validating Epoch_{self.epoch}")

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                inputs = self.inputs_to_device(inputs)

                outputs, losses = self.process_batch(inputs)

                mask = inputs["sphere_valid_mask"]
                pred_sem = outputs["pred_sem"].detach()
                gt_sem = inputs["sphere_gt_sem"]

                self.evaluator.compute_eval_metrics(gt_sem, pred_sem, mask, track=True)

        # 聚合 confusion matrix
        if self.distributed:
            cm = torch.tensor(self.evaluator.confusion_matrix.confusion_matrix, device=self.device)
            dist.all_reduce(cm, op=dist.ReduceOp.SUM)
            self.evaluator.confusion_matrix.confusion_matrix = cm.cpu().numpy().astype(np.int64)

        if not self.distributed or self.rank == 0:
            errors, best_errors = self.evaluator.get_results(update_best=True)
            self.evaluator.print()
            self.log("val", inputs, outputs, losses, errors, best_errors)

            # 保存当前最佳 mIoU 模型（只在主进程）
            current_best = best_errors.get("acc/iou", float("-inf"))
            if current_best > self.best_iou:
                self.best_iou = current_best
                self.save_model(is_best=True)

            del errors
        del inputs, outputs, losses

    def process_batch(self, inputs):
        x = inputs["normalized_sphere_rgb"]
        mask = inputs["sphere_valid_mask"]
        gt = inputs["sphere_gt_sem"]

        pred = self.model(x)

        loss_rec = self.compute_loss(pred.permute(0, 2, 1), gt.long())
        loss_eq = torch.tensor(0.0, device=pred.device)
        if self.eq_loss_weight > 0 and self.model.training:
            loss_eq = self.compute_eq_loss(x, pred)

        losses = {
            "loss": loss_rec + self.eq_loss_weight * loss_eq,
            "loss_rec": loss_rec,
            "loss_eq": loss_eq,
        }

        outputs = {
            "pred_sem": pred.detach(),
        }

        return outputs, losses

    def compute_eq_loss(self, x, pred):
        model = self.model.module if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else self.model
        x0 = model.project_input(x)

        if self.xyz_proj is None or self.xyz_img is None:
            proj_rank = model.proj_rank
            img_rank = model.img_rank
            ref = model.icosphere_ref
            self.xyz_proj = torch.tensor(ref.get_normals(proj_rank), dtype=torch.float32)
            self.xyz_img = torch.tensor(ref.get_normals(img_rank), dtype=torch.float32)

        eq_losses = []
        for _ in range(self.eq_loss_samples):
            R = self.random_rotation_matrix(device=x0.device)
            idx_proj = self.rotate_nodes_idx(self.xyz_proj.to(x0.device), R)
            idx_img = self.rotate_nodes_idx(self.xyz_img.to(pred.device), R)

            x0_rot = x0.index_select(1, idx_proj)
            logits_rot1 = model.forward_tokens(x0_rot)
            logits_rot2 = pred.index_select(1, idx_img).detach()

            eq_losses.append(F.mse_loss(logits_rot1, logits_rot2))

        return torch.stack(eq_losses).mean()

    def random_rotation_matrix(self, device):
        u1, u2, u3 = torch.rand(3, device=device)
        q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
        q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
        q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
        q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
        x, y, z, w = q1, q2, q3, q4
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ], dim=0)
        return R

    def rotate_nodes_idx(self, xyz, R, chunk_size=1024):
        # xyz: [N,3], R: [3,3]
        N = xyz.shape[0]
        rot = xyz @ R.T
        idx_all = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dots = rot[start:end] @ xyz.T
            idx = dots.argmax(dim=1)
            idx_all.append(idx)
        return torch.cat(idx_all, dim=0)

    def log(self, mode, inputs, outputs, losses, errors, best_errors=None):
        """Write an event to the tensorboard events file
        """

        # 仅主进程记录
        if self.distributed and self.rank != 0:
            return

        if self.wandb_run:
            wandb.log({f"losses_{mode}/{loss_key}": loss_val
                       for loss_key, loss_val in losses.items()
                       },
                      step=self.step)

            wandb.log({f"{key.split('/')[0]}_{mode}/{key.split('/')[1]}": val
                       for key, val in errors.items()
                       },
                      step=self.step)

            if best_errors is not None:
                wandb.log({f"best_{key.split('/')[0]}_{mode}/{key.split('/')[1]}": val
                           for key, val in best_errors.items()
                           },
                          step=self.step)

        if self.log_file:
            # 将当前指标和损失写入本地日志
            lines = []
            for k, v in losses.items():
                lines.append(f"{k}={float(v):.6f}")
            for k, v in errors.items():
                lines.append(f"{k}={float(v):.6f}")
            if best_errors is not None:
                for k, v in best_errors.items():
                    lines.append(f"best_{k}={float(v):.6f}")
            with open(self.log_file, "a") as f:
                f.write(f"{mode},step={self.step},epoch={self.epoch}," + " ".join(lines) + "\n")

    def save_model(self, is_best: bool = False):
        """Save model weights to disk
        """
        if self.distributed and self.rank != 0:
            return

        if self.wandb_run:
            save_folder = os.path.join(self.wandb_run.dir, "models")
        else:
            save_folder = os.path.join(self.args.log_dir, "models")
        os.makedirs(save_folder, exist_ok=True)

        print(f"Saving model at {save_folder}" + (" (best mIoU)" if is_best else ""))

        suffix = "model_best_miou" if is_best else "model"
        model_save_path = os.path.join(save_folder, f"{suffix}.pth")
        model_to_save = self.model
        if isinstance(model_to_save, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_to_save = model_to_save.module
        model_state_dict = model_to_save.state_dict()
        torch.save(model_state_dict, model_save_path)

        # 仅对“最新”模型保存优化器和上传，以避免覆盖最佳模型文件
        if not is_best:
            opt_save_path = os.path.join(save_folder, "{}.pth".format("optimizer"))
            opt_state_dict = self.optimizer.state_dict()
            torch.save(opt_state_dict, opt_save_path)

            if self.wandb_run:
                artifact = wandb.Artifact(name=f"model-{self.wandb_run.id}", type="model", metadata=self.config)
                artifact.add_file(model_save_path)
                self.wandb_run.log_artifact(artifact)

                artifact = wandb.Artifact(name=f"optimizer-{self.wandb_run.id}", type="optimizer", metadata={})
                artifact.add_file(opt_save_path)
                self.wandb_run.log_artifact(artifact)

    def load_model(self):
        """Load model from disk
        """
        load_run = wandb.Api().run(f"{self.args.wandb_entity}/{self.args.wandb_project}/{self.args.load_weights_task}")

        artifacts = load_run.logged_artifacts()

        model_art = [art for art in artifacts if art.type == "model" and "latest" in art.aliases]
        assert len(model_art) == 1, f"Loaded weights task should have 1 latest model, got {len(model_art)}"
        opt_art = [art for art in artifacts if art.type == "optimizer" and "latest" in art.aliases]
        assert len(opt_art) <= 1, f"Loaded weights task should at most 1 latest optimizer, got {len(opt_art)}"

        model_dir = model_art[0].download(f"{self.args.log_dir}/PRETRAINED/{load_run.id}")
        if len(opt_art):
            opt_dir = opt_art[0].download(f"{self.args.log_dir}/PRETRAINED/{load_run.id}")

        print(f"Loading Model weights from {model_dir}")
        pretrained_dict = torch.load(os.path.join(model_dir, "model.pth"))
        model_to_load = self.model
        if isinstance(model_to_load, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_to_load = model_to_load.module
        missing_keys, unexpected_keys = model_to_load.load_state_dict(pretrained_dict, strict=False)
        if len(missing_keys):
            warnings.warn(f"MISSING KEYS : {missing_keys}")
        assert len(unexpected_keys) == 0, f"{unexpected_keys}"

        if len(opt_art) and hasattr(self, "optimizer"):
            print("Loading Optimizer weights")
            optimizer_dict = torch.load(os.path.join(opt_dir, "optimizer.pth"))
            self.optimizer.load_state_dict({k: v for k, v in optimizer_dict.items() if k not in missing_keys})
        else:
            print("Optimizer weights were not saved")

    def load_model_from_path(self, weight_path: str):
        """Load model weights from a local path (no wandb)
        """
        print(f"Loading model weights from {weight_path}")
        state = torch.load(weight_path, map_location=self.device)
        model_to_load = self.model
        if isinstance(model_to_load, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_to_load = model_to_load.module
        missing_keys, unexpected_keys = model_to_load.load_state_dict(state, strict=False)
        if len(missing_keys):
            warnings.warn(f"MISSING KEYS : {missing_keys}")
        if len(unexpected_keys):
            warnings.warn(f"UNEXPECTED KEYS : {unexpected_keys}")
