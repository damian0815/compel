from dataclasses import dataclass

import torch

__all__ = ["ConditioningScheduler", "Conditioning", "StaticConditioningScheduler"]

@dataclass
class Conditioning:
    """
    Conditioning. In all examples `B` is batch size, `77` is the text encoder's max token length,
    and `token_dim` is 768 for SD1 and 1280 for SD2.
    """
    positive_conditioning: torch.Tensor # shape [B x 77 x token_dim]
    negative_conditioning: torch.Tensor # shape [B x 77 x token_dim]


class ConditioningScheduler:
    """
    Provides a mechanism to control which processes to apply for any given step of a Stable Diffusion generation.
    """

    def get_conditioning_for_step_pct(self, step_pct: float) -> Conditioning:
        """
        Return the conditioning to apply at the given step.
        :param step_pct: The step as a float `0..1`, where `0.0` is immediately before the start of image generation
        process (when the latent vector is 100% noise), and `1.0` is immediately after the end of the final step
        (when the latent vector represents the final noise-free generated image).
        :return: The Conditioning to apply for the requested step.
        """
        raise NotImplementedError("Subclasses must override")


class StaticConditioningScheduler(ConditioningScheduler):
    def __init__(self, positive_conditioning: torch.Tensor,
                 negative_conditioning: torch.Tensor):
        self.positive_conditioning = positive_conditioning
        self.negative_conditioning = negative_conditioning

    def get_conditioning_for_step_pct(self, step_pct: float) -> Conditioning:
        """ See base class for docs. """
        return Conditioning(positive_conditioning=self.positive_conditioning,
                            negative_conditioning=self.negative_conditioning,
                            )
