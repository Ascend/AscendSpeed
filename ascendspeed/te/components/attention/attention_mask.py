from typing import Optional, Type, TypeVar

import torch
from ascendspeed.error_utils import check_equal

Self = TypeVar("Self", bound="AttentionMask")


class AttentionMask:
    """
    Holds an attention mask, along with a couple of helpers and attributes.

    .. note: this is an additive mask, meaning that coefficients which should be computed hold the '0.' value,
        and coefficients which should be skipped hold the '-inf' value. Any other value is possible if the purpose
        is to bias the attention computation for instance

    .. note: the attention mask dimensions are expected to be `[batch, to_sequence, from_sequence]`,
        `[to_sequence, from_sequence]`, or anything broadcastable in between
    """

    def __init__(self, additive_mask: torch.Tensor, is_causal: bool = False):
        if additive_mask.requires_grad:
            raise Exception("additive_mask doesn't need grad info")

        if additive_mask.ndim == 2:
            additive_mask = additive_mask.unsqueeze(0)

        self.values = additive_mask
        self.is_causal = is_causal
        self.seq_len = additive_mask.shape[1]
        self.to_seq_len = additive_mask.shape[0]

    def to_bool(self) -> torch.Tensor:
        """
        .. warning: we assume here that True implies that the value should be computed
        """
        return self.values != float("-inf")

    @classmethod
    def from_bool(cls: Type[Self], x: torch.Tensor) -> Self:
        """
        Create an AttentionMask given a boolean pattern.
        .. warning: we assume here that True implies that the value should be computed
        """
        check_equal(x.dtype, torch.bool, "input mask should be bool!")

        additive_mask = torch.empty_like(x, dtype=torch.float, device=x.device)
        additive_mask.masked_fill_(x, 0.0)
        additive_mask.masked_fill_(~x, float("-inf"))

        return cls(additive_mask)

    @classmethod
    def from_multiplicative(cls: Type[Self], x: torch.Tensor) -> Self:
        """
        Create an AttentionMask given a multiplicative attention mask.
        """
        check_equal(x.dtype, torch.bool, "input mask should be bool!")

        additive_mask = torch.empty_like(x, dtype=torch.float, device=x.device)
        x = x.bool()

        additive_mask.masked_fill_(x, 0.0)
        additive_mask.masked_fill_(~x, float("-inf"))

        return cls(additive_mask)

    @classmethod
    def make_causal(
        cls: Type[Self],
        seq_len: int,
        to_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        if not to_seq_len:
            to_seq_len = seq_len

        additive_mask = torch.triu(
            torch.ones(seq_len, to_seq_len, device=device, dtype=dtype) * float("-inf"),
            diagonal=1,
        )
        return cls(additive_mask=additive_mask, is_causal=True)

    def make_crop(
        self, seq_len: int, to_seq_len: Optional[int] = None
    ) -> "AttentionMask":
        """
        Return a cropped attention mask, whose underlying tensor is a view of this one
        """

        if not to_seq_len:
            to_seq_len = seq_len

        return AttentionMask(
            self.values[:, :seq_len, :to_seq_len], is_causal=self.is_causal
        )

    def __repr__(self):
        return f"AttentionMask - causal {self.is_causal} - mask " + str(self.values)

    @property
    def device(self):
        return self.values.device

    @property
    def is_sparse(self):
        return False

    @property
    def ndim(self):
        return len(self.values.shape)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    def __add__(self, other):
        return AttentionMask(self.values + other.values, is_causal=False)

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "AttentionMask":
        if device is not None and not isinstance(device, torch.device):
            raise Exception("invalid device.")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise Exception("invald dtype.")
        if device is None and dtype is None:
            raise Exception("device and dtype are both None.")

        # Noop if we don't need to create another instance
        proper_device = not device or (device and device == self.device)
        proper_dtype = not dtype or (dtype and dtype == self.dtype)
        if proper_device and proper_dtype:
            return self

        return AttentionMask(self.values.to(device=device, dtype=dtype), self.is_causal)
