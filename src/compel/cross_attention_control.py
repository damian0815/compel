import torch
from typing import List


class Arguments:
    def __init__(self,
                 original_conditioning: torch.Tensor,
                 edited_conditioning: torch.Tensor,
                 edit_opcodes: List[tuple],
                 edit_options: List[dict]):
        """
        Arguments for a cross-attention control implementation that substitutes `edited_conditioning` for `original_conditioning` while applying the
        attention maps from `original_conditioning`.

        :param edit_opcodes: a list of difflib.SequenceMatcher-like opcodes describing how to map original conditioning tokens to edited conditioning tokens (only the 'equal' opcode is required)
        :param edit_options: if doing cross-attention control, per-edit options. there should be 1 item in edit_options for each item in edit_opcodes.
        """
        # todo: rewrite this to take embedding fragments rather than a single edited_conditioning vector
        self.original_conditioning = original_conditioning
        self.edited_conditioning = edited_conditioning
        self.edit_opcodes = edit_opcodes

        assert len(edit_opcodes) == len(edit_options), \
                "there must be 1 edit_options dict for each edit_opcodes tuple"
        non_none_edit_options = [x for x in edit_options if x is not None]
        assert len(non_none_edit_options)>0, "missing edit_options"
        if len(non_none_edit_options)>1:
            print('warning: cross-attention control options are not working properly for >1 edit')
        self.edit_options = non_none_edit_options[0]

