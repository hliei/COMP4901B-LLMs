"""Utility functions for student-implemented loss computations.

The training entry point expects a callable named `compute_loss_from_logits`.
Students should implement the function so that it takes model logits and
ground truth labels and returns a scalar loss tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother

from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    labels: Optional[torch.Tensor],
    num_items_in_batch: int,
) -> torch.Tensor:
    """Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        ignore_index: Label id that should be ignored when computing the loss. The
            trainer passes HuggingFace's default ignore index (-100).

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.

    Students should implement this function by computing the cross-entropy loss
    from the raw logits. You may not call `torch.nn.CrossEntropyLoss`; instead,
    derive the loss explicitly using a log-softmax over the vocabulary dimension.
    """

    # raise NotImplementedError("Implement token-level cross-entropy using the logits.")
    logits = outputs.logits
    return cross_entropy_loss(logits, labels, num_items_in_batch=num_items_in_batch)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
) -> torch.Tensor:
    """
    Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        num_items_in_batch: Number of valid items in batch for normalization.

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.
    """
    # ----- Implementation of cross-entropy loss -----
    shift_logits = logits[:, : -1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    mask = (flat_labels != IGNORE_TOKEN_ID)
    valid_logits = flat_logits[mask]
    valid_labels = flat_labels[mask]

    log_probs = F.log_softmax(valid_logits, dim=-1)
    selected_log_probs = log_probs[range(valid_labels.size(0)), valid_labels]

    neg_log_likelihood = -selected_log_probs
    total_loss = neg_log_likelihood.sum()
    mean_loss = total_loss / num_items_in_batch
    
    return mean_loss