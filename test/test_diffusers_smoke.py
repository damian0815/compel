import gc
import os
import unittest
from pathlib import Path

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from compel import CompelForSD, CompelForSDXL

try:
    from prompting_test_utils import DummyTokenizer
except ModuleNotFoundError:
    from test.prompting_test_utils import DummyTokenizer


LOCAL_SDXL_CHECKPOINTS = {
    "sd_xl_turbo": Path("/mnt/s/Code/Models/ImageGen/CUI-Archived/checkpoints/sd_xl_turbo_1.0_fp16.safetensors"),
    "cyberrealisticXL_v80": Path("/mnt/s/Code/Models/ImageGen/CUI-Archived/checkpoints/cyberrealisticXL_v80_fp16.safetensors"),
}
RUN_LOCAL_CHECKPOINT_TESTS = os.getenv("COMPEL_RUN_LOCAL_CHECKPOINT_TESTS") == "1"


def make_clip_text_config(hidden_size: int = 32, projection_dim: int = 32, max_position_embeddings: int = 16) -> CLIPTextConfig:
    return CLIPTextConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=37,
        projection_dim=projection_dim,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=10,
        pad_token_id=11,
        eos_token_id=12,
    )


def make_tiny_vae() -> AutoencoderKL:
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D"],
        block_out_channels=[32],
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
    )


def make_tiny_sd_unet(cross_attention_dim: int) -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=(4, 8),
        norm_num_groups=8,
    )


def make_tiny_sdxl_unet(cross_attention_dim: int, projection_dim: int) -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=(4, 8),
        norm_num_groups=8,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=(8 * 6) + projection_dim,
    )


class DiffusersSmokeTestCase(unittest.TestCase):
    def test_stable_diffusion_pipeline_accepts_compel_prompt_embeds(self):
        tokenizer = DummyTokenizer(model_max_length=16)
        text_encoder = CLIPTextModel(make_clip_text_config(hidden_size=32, projection_dim=32, max_position_embeddings=16))
        pipe = StableDiffusionPipeline(
            vae=make_tiny_vae(),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=make_tiny_sd_unet(cross_attention_dim=32),
            scheduler=EulerDiscreteScheduler(num_train_timesteps=10, steps_offset=1),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to("cpu")
        pipe.set_progress_bar_config(disable=True)

        conditioning = CompelForSD(pipe)("a b c", negative_prompt="a")
        result = pipe(
            prompt_embeds=conditioning.embeds,
            negative_prompt_embeds=conditioning.negative_embeds,
            num_inference_steps=1,
            guidance_scale=2.0,
            output_type="np",
        )

        self.assertEqual(conditioning.embeds.shape, (1, 16, 32))
        self.assertEqual(conditioning.negative_embeds.shape, (1, 16, 32))
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.images[0].shape, (32, 32, 3))

    def test_sdxl_pipeline_accepts_compel_prompt_and_pooled_embeds(self):
        tokenizer = DummyTokenizer(model_max_length=16)
        text_encoder = CLIPTextModel(make_clip_text_config(hidden_size=32, projection_dim=16, max_position_embeddings=16))
        text_encoder_2 = CLIPTextModelWithProjection(
            make_clip_text_config(hidden_size=32, projection_dim=16, max_position_embeddings=16)
        )
        pipe = StableDiffusionXLPipeline(
            vae=make_tiny_vae(),
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=DummyTokenizer(model_max_length=16),
            unet=make_tiny_sdxl_unet(cross_attention_dim=64, projection_dim=16),
            scheduler=EulerDiscreteScheduler(num_train_timesteps=10, steps_offset=1),
            image_encoder=None,
            feature_extractor=None,
        ).to("cpu")
        pipe.set_progress_bar_config(disable=True)

        conditioning = CompelForSDXL(pipe)(
            "a b c",
            style_prompt="b c",
            negative_prompt="a",
            negative_style_prompt="a",
        )
        result = pipe(
            prompt_embeds=conditioning.embeds,
            pooled_prompt_embeds=conditioning.pooled_embeds,
            negative_prompt_embeds=conditioning.negative_embeds,
            negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
            num_inference_steps=1,
            guidance_scale=2.0,
            output_type="np",
        )

        self.assertEqual(conditioning.embeds.shape, (1, 16, 64))
        self.assertEqual(conditioning.pooled_embeds.shape, (1, 16))
        self.assertEqual(conditioning.negative_embeds.shape, (1, 16, 64))
        self.assertEqual(conditioning.negative_pooled_embeds.shape, (1, 16))
        self.assertEqual(len(result.images), 1)
        self.assertEqual(result.images[0].shape, (32, 32, 3))


@unittest.skipUnless(RUN_LOCAL_CHECKPOINT_TESTS, "set COMPEL_RUN_LOCAL_CHECKPOINT_TESTS=1 to run local checkpoint smoke tests")
class LocalCheckpointSmokeTestCase(unittest.TestCase):
    def test_sdxl_single_file_checkpoints_load_locally(self):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        for checkpoint_name, checkpoint_path in LOCAL_SDXL_CHECKPOINTS.items():
            with self.subTest(checkpoint=checkpoint_name):
                self.assertTrue(checkpoint_path.exists(), f"missing local checkpoint: {checkpoint_path}")

                pipe = StableDiffusionXLPipeline.from_single_file(
                    str(checkpoint_path),
                    local_files_only=True,
                    torch_dtype=dtype,
                )
                pipe.set_progress_bar_config(disable=True)

                self.assertIsInstance(pipe, StableDiffusionXLPipeline)
                self.assertIsNotNone(pipe.text_encoder)
                self.assertIsNotNone(pipe.text_encoder_2)
                self.assertGreater(pipe.unet.config.cross_attention_dim, 0)

                del pipe
                gc.collect()


if __name__ == "__main__":
    unittest.main()
