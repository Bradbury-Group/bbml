"""Qwen Image Foundation - bbml Foundation wrapper for Qwen Image Edit model."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from loguru import logger
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim.optimizer import ParamsT

from bbml import print_gpu_memory, texam
from bbml.core.data_transform import DataTransform
from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.foundation import Foundation

from .data_transforms import (
    QwenImageDataTransform,
    QwenReferenceDataTransform,
    QwenTextDataTransform,
)
from .datamodels import QwenConfig, QwenInput, QwenOutput
from .models.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
    calculate_dimensions,
)
from .models.transformer_qwenimage import QwenImageTransformer2DModel
from .sampling import TimestepDistUtils


class QwenImageFoundation(Foundation):
    """Foundation wrapper for Qwen Image Edit diffusion model.

    Implements bbml Foundation interface for training and inference with
    Qwen's dual-stream DiT architecture for image editing.

    Args:
        config: Model configuration.
        train_config: Training configuration (optional for inference).
    """

    SOURCE = "Qwen/Qwen-Image-Edit-2509"
    serialize_modules = ["transformer"]

    def __init__(
        self,
        config: QwenConfig,
        train_config: TrainerConfig | None = None,
    ):
        super().__init__(config, train_config)
        self.config: QwenConfig = config

        self.dtype = torch.bfloat16
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = default_device
        logger.info(f"Initializing QwenImageFoundation on {self.device}")

        # Load pipeline with custom transformer
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            config.from_pretrained,
            transformer=QwenImageTransformer2DModel.from_pretrained(
                config.from_pretrained,
                subfolder="transformer",
                torch_dtype=self.dtype,
                device_map=self.device,
            ),
            torch_dtype=self.dtype,
        )
        pipe = pipe.to(device=self.device, dtype=self.dtype)

        # Optional multi-view LoRA
        if config.load_multi_view_lora:
            pipe.load_lora_weights(
                "dx8152/Qwen-Edit-2509-Multiple-angles",
                weight_name="镜头转换.safetensors",
                adapter_name="angles",
            )
            pipe.set_adapters(["angles"], adapter_weights=[1.0])
            pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
            pipe.unload_lora_weights()

        self.pipe = pipe
        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler

        # Freeze non-trainable components
        self.vae.to(self.dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder_device = None
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        if config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        if config.vae_tiling:
            self.vae.enable_tiling(576, 576, 512, 512)

        # Initialize timestep utilities
        self.timestep_dist_utils = TimestepDistUtils(
            min_seq_len=self.scheduler.config.base_image_seq_len,
            max_seq_len=self.scheduler.config.max_image_seq_len,
            min_mu=self.scheduler.config.base_shift,
            max_mu=self.scheduler.config.max_shift,
            train_dist=config.train_dist,
            train_shift=config.train_shift,
            inference_dist=config.inference_dist,
            inference_shift=config.inference_shift,
            static_mu=config.static_mu,
            loss_weight_dist=config.loss_weight_dist,
        )

    # -------------------------------------------------------------------------
    # Trainable interface
    # -------------------------------------------------------------------------

    def single_step(self, batch: dict[str, Any]) -> Tensor:
        """Execute one training step with flow matching.

        Args:
            batch: Dict with 'text', 'reference', 'image' keys.

        Returns:
            Scalar loss tensor.
        """
        self._offload_text_encoder("cpu")

        # Encode prompts if not cached
        if "prompt_embeds" not in batch:
            batch = self._preprocess_batch(batch)

        prompt_embeds, prompt_embeds_mask = batch["prompt_embeds"]
        prompt_embeds = prompt_embeds.to(device=self.device)
        prompt_embeds_mask = prompt_embeds_mask.to(device=self.device)

        # Get target and noise latents
        images = batch["image"].to(device=self.device, dtype=self.dtype)
        x_0 = self._pil_to_latents(images)
        x_1 = torch.randn_like(x_0)

        # Sample timestep
        seq_len = self.timestep_dist_utils.get_seq_len(x_0)
        batch_size = x_0.shape[0]
        t = self.timestep_dist_utils.get_train_t([batch_size], seq_len=seq_len)
        t = t.to(device=self.device, dtype=self.dtype)

        # Flow matching interpolation
        x_t = (1.0 - t) * x_0 + t * x_1
        x_t_1d = self._pack_latents(x_t)

        # Encode reference
        references = batch["reference"].to(device=self.device, dtype=self.dtype)
        refs = self._pil_to_latents(references)
        refs_1d = self._pack_latents(refs)

        # Concatenate target and reference latents
        inp_1d = torch.cat([x_t_1d, refs_1d], dim=1)

        # Compute RoPE embeddings
        l_height, l_width = x_0.shape[-2:]
        ref_height, ref_width = refs.shape[-2:]
        img_shapes = [
            [
                (1, l_height // 2, l_width // 2),
                (1, ref_height // 2, ref_width // 2),
            ]
        ] * batch_size
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        image_rotary_emb = self.transformer.pos_embed(
            img_shapes, txt_seq_lens, device=x_0.device
        )

        # Forward pass
        v_pred_1d = self.transformer(
            hidden_states=inp_1d,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=t,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        # Extract prediction for target (not reference)
        v_pred_1d = v_pred_1d[:, : x_t_1d.size(1)]
        v_pred_2d = self._unpack_latents(v_pred_1d, h=l_height // 2, w=l_width // 2)

        # Velocity target
        v_gt_2d = x_1 - x_0

        # Compute loss with optional weighting
        if self.config.loss_weight_dist is not None:
            loss = F.mse_loss(v_pred_2d, v_gt_2d, reduction="none").mean(dim=[1, 2, 3])
            weights = self.timestep_dist_utils.get_loss_weighting(t)
            loss = torch.mean(loss * weights)
        else:
            loss = F.mse_loss(v_pred_2d, v_gt_2d, reduction="mean")

        return loss

    def get_train_parameters(self) -> ParamsT:
        """Return parameters for optimizer.

        Returns:
            List of parameter groups with weight decay separation.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            # Weight decay for 2D+ tensors, no decay for 1D (biases, norms)
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    @property
    def data_transforms(self) -> dict[str, DataTransform]:
        """Return data transforms for each batch field."""
        return {
            "image": QwenImageDataTransform(self.config.vae_image_size),
            "reference": QwenReferenceDataTransform(CONDITION_IMAGE_SIZE),
            "text": QwenTextDataTransform(),
        }

    # -------------------------------------------------------------------------
    # Runnable interface
    # -------------------------------------------------------------------------

    @property
    def input_model(self) -> type[QwenInput]:
        return QwenInput

    @property
    def output_model(self) -> type[QwenOutput]:
        return QwenOutput

    def run(self, input: QwenInput) -> QwenOutput:
        """Run inference.

        Args:
            input: Inference input with image, prompt, etc.

        Returns:
            Generated images.
        """
        self._offload_text_encoder("cuda")

        if input.vae_image_override is None:
            input.vae_image_override = self.config.vae_image_size
        if input.latent_size_override is None:
            input.latent_size_override = self.config.vae_image_size

        result = self.pipe(**input.model_dump())
        return QwenOutput(images=result.images)

    # -------------------------------------------------------------------------
    # Serializable interface
    # -------------------------------------------------------------------------

    def save(self, save_path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            save_path: Directory to save to.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        for module_name in self.serialize_modules:
            module = getattr(self, module_name)
            state_dict = {k: v.cpu() for k, v in module.state_dict().items()}
            save_file(state_dict, save_path / f"{module_name}.safetensors")
            logger.info(f"Saved {module_name} to {save_path}")

    def load(self, load_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            load_path: Directory to load from.
        """
        load_path = Path(load_path)
        if not load_path.is_dir():
            raise ValueError(f"Expected {load_path=} to be a directory")

        for module_name in self.serialize_modules:
            ckpt_path = load_path / f"{module_name}.safetensors"
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue

            state_dict = load_file(str(ckpt_path))
            module = getattr(self, module_name)
            missing, unexpected = module.load_state_dict(
                state_dict, strict=False, assign=True
            )
            if missing:
                warnings.warn(f"{module_name} missing keys: {missing}")
            if unexpected:
                warnings.warn(f"{module_name} unexpected keys: {unexpected}")
            logger.info(f"Loaded {module_name} from {ckpt_path}")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _offload_text_encoder(self, device: str | torch.device) -> None:
        """Move text encoder to specified device."""
        if self.text_encoder_device == device:
            return
        logger.debug(f"Moving text encoder to {device}")
        self.text_encoder_device = device
        self.text_encoder.to(device)
        if device == "cpu" or device == torch.device("cpu"):
            print_gpu_memory(clear_mem="pre")

    def _preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Encode text prompts for batch.

        Args:
            batch: Batch with 'text' and 'reference' keys.

        Returns:
            Batch with added 'prompt_embeds' key.
        """
        prompts = batch["text"]
        references = batch["reference"]

        h, w = references.shape[-2:]
        h_r, w_r = calculate_dimensions(CONDITION_IMAGE_SIZE, h / w)
        references = TF.resize(references, (h_r, w_r))

        self._offload_text_encoder("cuda")

        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
                prompts,
                references.mul(255),  # Scale to RGB [0, 255]
                device="cuda",
                max_sequence_length=self.config.train_max_sequence_length,
            )

        prompt_embeds = prompt_embeds.cpu().clone().detach()
        prompt_embeds_mask = prompt_embeds_mask.cpu().clone().detach()

        batch["prompt_embeds"] = (prompt_embeds, prompt_embeds_mask)
        batch["reference"] = batch["reference"].cpu()
        batch["image"] = batch["image"].cpu()

        return batch

    def _pil_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latents.

        Args:
            images: Image tensor [B, C, H, W] in [0, 1].

        Returns:
            Normalized latents [B, C, H/8, W/8].
        """
        image = self.pipe.image_processor.preprocess(images)

        h, w = image.shape[-2:]
        h_r, w_r = calculate_dimensions(self.config.vae_image_size, h / w)
        image = TF.resize(image, (h_r, w_r))

        image = image.unsqueeze(2)  # [B, C, F=1, H, W]
        image = image.to(device=self.device, dtype=self.dtype)
        latents = self.pipe.vae.encode(image).latent_dist.mode()

        # Normalize latents
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std
        latents = latents.squeeze(2)

        return latents.to(dtype=self.dtype)

    def _latents_to_pil(
        self, latents: torch.Tensor, h: int | None = None, w: int | None = None
    ) -> list[Image.Image]:
        """Decode latents to PIL images.

        Args:
            latents: Latent tensor (packed 1D or unpacked 2D).
            h: Height for unpacking (if latents are 1D).
            w: Width for unpacking (if latents are 1D).

        Returns:
            List of PIL images.
        """
        latents = latents.clone().detach()

        if latents.dim() == 3:  # Packed 1D latent
            if h is None or w is None:
                raise ValueError("Auto unpack needs h, w")
            latents = self._unpack_latents(latents, h=h, w=w)

        latents = latents.unsqueeze(2)
        latents = latents.to(self.dtype)

        # Denormalize
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        latents = latents.to(device=self.device, dtype=self.dtype)
        image = self.pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return self.pipe.image_processor.postprocess(image)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack 2D latents into 1D sequence.

        Args:
            latents: [B, C, H, W] tensor.

        Returns:
            [B, H*W/4, C*4] packed tensor.
        """
        # Pack 2x2 patches: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
        return rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def _unpack_latents(packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Unpack 1D sequence to 2D latents.

        Args:
            packed: [B, L, C*4] packed tensor.
            h: Target height (in latent space / 2).
            w: Target width (in latent space / 2).

        Returns:
            [B, C, H, W] unpacked tensor.
        """
        return rearrange(
            packed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=h, w=w
        )
