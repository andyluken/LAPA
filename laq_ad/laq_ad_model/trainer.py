"""Training loop for LAQ-AD.

Uses HuggingFace accelerate for multi-GPU support — same dependency as the
upstream laq trainer, so no new requirements. The training loop is deliberately
simpler than laq_trainer.py:
  - No EMA (FSQ doesn't need it)
  - No NSVQ replace_unused_codebooks schedule (FSQ doesn't collapse)
  - Explicit can_bus loss handling and logging
  - Saves model state_dict (not full trainer checkpoint) for portability

Usage:
    trainer = LAQADTrainer(
        model=laq_ad_model,
        dataset=dataset,
        sampler=maneuver_balanced_sampler(dataset),
        results_folder="results_laq_ad",
        num_train_steps=50_000,
        batch_size=16,
        lr=1e-4,
    )
    trainer.train()
"""

import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import LAQADModel


class LAQADTrainer:
    """Accelerate-backed training loop for LAQ-AD.

    Args:
        model:              LAQADModel instance (un-moved to device; trainer handles that)
        dataset:            NuScenesLAQDataset
        sampler:            Optional WeightedRandomSampler (from maneuver_balanced_sampler)
        results_folder:     Directory for checkpoint saves and sample reconstructions
        num_train_steps:    Total gradient steps
        batch_size:         Per-GPU batch size
        grad_accum_every:   Gradient accumulation steps (effective_batch = batch_size * grad_accum)
        lr:                 Peak learning rate (cosine schedule with linear warmup)
        warmup_steps:       LR warmup steps
        save_model_every:   Checkpoint interval in steps
        save_results_every: Log interval in steps (also saves reconstruction samples)
        use_wandb:          Enable WandB logging
        collate_fn:         Custom collate for variable-None tensors in batch
    """

    def __init__(
        self,
        model: LAQADModel,
        dataset,
        sampler: Optional[Sampler] = None,
        results_folder: str = "results_laq_ad",
        num_train_steps: int = 50_000,
        batch_size: int = 16,
        grad_accum_every: int = 1,
        lr: float = 1e-4,
        warmup_steps: int = 1_000,
        save_model_every: int = 5_000,
        save_results_every: int = 500,
        use_wandb: bool = False,
        collate_fn=None,
        resume_from: Optional[str] = None,
    ):
        if not ACCELERATE_AVAILABLE:
            raise ImportError("pip install accelerate")

        from .data import collate_fn as default_collate
        self.accelerator = Accelerator(gradient_accumulation_steps=grad_accum_every)

        self.model = model
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.use_wandb = use_wandb and WANDB_AVAILABLE and self.accelerator.is_main_process

        _collate = collate_fn or default_collate
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=_collate,
            num_workers=4,
            pin_memory=True,
        )

        self.optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
        self.scheduler = self._cosine_schedule_with_warmup(
            self.optimizer, warmup_steps=warmup_steps, total_steps=num_train_steps
        )

        (self.model, self.optimizer, self.dataloader, self.scheduler) = (
            self.accelerator.prepare(self.model, self.optimizer, self.dataloader, self.scheduler)
        )

        self.step = 0
        self._resume_step = 0   # used to skip the checkpoint-overwrite on first step
        if resume_from is not None:
            ckpt = Path(resume_from)
            if ckpt.exists():
                self.accelerator.unwrap_model(self.model).load(ckpt)
                # Parse step from filename laq_ad.NNNNN.pt
                self.step = int(ckpt.stem.split(".")[-1])
                self._resume_step = self.step
                # Fast-forward LR scheduler so the cosine schedule is correct.
                # LambdaLR.step() is pure arithmetic — 50K calls takes ~milliseconds.
                for _ in range(self.step):
                    self.scheduler.step()
                print(f"  Resumed from {ckpt} at step {self.step}, "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e}")
            else:
                print(f"  WARNING: resume_from={resume_from} not found — starting fresh")

    @staticmethod
    def _cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lr_lambda)

    def _save_config(self, model):
        """Write model config to results_folder/config.json so eval can auto-detect architecture."""
        import json
        cfg = {
            "model_type": type(model).__name__,
            "levels": model.fsq.levels,
            "dim": model.project_out.out_features,
            "image_size": list(model.image_size),
            "patch_size": list(model.patch_size),
            "heatmap_alpha": model.heatmap_alpha,
            "can_bus_weight": model.can_bus_weight,
            "entropy_reg_weight": model.entropy_reg_weight,
            "spread_reg_weight": model.spread_reg_weight,
            "covariance_reg_weight": model.covariance_reg_weight,
            "flow_prediction_weight": model.flow_prediction_weight,
            "recon_loss_weight": model.recon_loss_weight,
        }
        # SigLIP-specific fields
        if hasattr(model, "siglip_model_name"):
            cfg["siglip_model_name"] = model.siglip_model_name
        with open(self.results_folder / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    def _data_iter(self):
        """Infinite iterator over the dataloader."""
        while True:
            yield from self.dataloader

    def train(self):
        self.model.train()
        log = self.accelerator.log if hasattr(self.accelerator, "log") else (lambda *a, **k: None)

        if self.use_wandb:
            wandb.watch(self.model, log_freq=self.save_results_every)

        data_iter = self._data_iter()

        while self.step < self.num_train_steps:
            with self.accelerator.accumulate(self.model):
                video, heatmap, can_bus, _ = next(data_iter)

                # Move optional tensors to device (DataLoader pin_memory only works for Tensors)
                if heatmap is not None:
                    heatmap = heatmap.to(self.accelerator.device)
                if can_bus is not None:
                    can_bus = can_bus.to(self.accelerator.device)

                loss, n_unique, aux = self.model(video, heatmap=heatmap, can_bus=can_bus)

                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.is_main_process and self.step % self.save_results_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"step {self.step:>6} | loss {loss.item():.4f} | "
                    f"recon {aux['recon_loss']:.4f} | can {aux['can_bus_loss']:.4f} | "
                    f"flow {aux.get('flow_loss', 0.0):.4f} | "
                    f"spread {aux.get('spread_loss', 0.0):.3f} | "
                    f"cov {aux.get('cov_loss', 0.0):.3f} | "
                    f"H_neg {aux.get('entropy_neg', 0.0):.3f} | "
                    f"perplexity {aux['perplexity']:.2f} | codes {n_unique} | lr {lr:.2e}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "recon_loss": aux["recon_loss"],
                            "can_bus_loss": aux["can_bus_loss"],
                            "entropy_neg": aux.get("entropy_neg", 0.0),
                            "perplexity": aux["perplexity"],
                            "n_unique_codes": n_unique,
                            "lr": lr,
                            "step": self.step,
                        }
                    )

            if self.accelerator.is_main_process and self.step % self.save_model_every == 0 and self.step > 0 and self.step != self._resume_step:
                ckpt = self.results_folder / f"laq_ad.{self.step}.pt"
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.save(ckpt)
                self._save_config(unwrapped)
                print(f"  saved checkpoint → {ckpt}")

            self.step += 1

        # Save final checkpoint
        if self.accelerator.is_main_process:
            ckpt = self.results_folder / f"laq_ad.{self.step}.pt"
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save(ckpt)
            self._save_config(unwrapped)
            print(f"Training complete. Final checkpoint → {ckpt}")
