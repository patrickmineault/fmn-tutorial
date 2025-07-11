"""
A reimplementation of the NDT model from scratch.
This implementation is based on the original paper, and it was designed to:
* Have less cruft (e.g. vanilla transformer layers, no custom initialization, etc.)
* Be a single file
* Be easy to read
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_mask_to_attn_mask(x):
    """
    Converts a binary mask to an attention mask.
    Args:
        x (torch.Tensor): A binary mask tensor of shape (batch_size, seq_len).
    Returns:
        torch.Tensor: An attention mask tensor of shape (batch_size, seq_len) with -inf for masked positions and 0 for unmasked positions.
    """
    return x.float().masked_fill(x == 0, float("-inf")).masked_fill(x == 1, float(0.0))


class _SinusoidalPositionEmb(nn.Module):
    """
    Sinusoidal positional encoding as described in the Transformer paper.
    This encoding is used to inject positional information into the model.
    It generates a fixed encoding based on the position and dimension.
    """

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        self.register_buffer("pe", pe)

    def forward(self, t: int) -> torch.Tensor:  # (t, d_model)
        return self.pe[:t]


class TransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder for sequence reconstruction.
    This model uses a series of transformer encoder layers to process input sequences
    and reconstruct them in the output space.

    It is similar a reimplementation of the NDT model.
    It limits the number of knobs that can be changed, in order to enhance readability.

    It does have two big knobs:
    - `pos_encoding_type`: Type of positional encoding to use (learned, sinusoidal, or none).
    - `projection`: Type of input/output projection (linear, identity, tied, input_only, output_only).
       output_only is what is typically implemented in NDT-1 style models. linear is conceptually closer to NDT-1-stitch style models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 10_000,
        pos_encoding_type: str = "sin",  # "learned" | "sin" | "none"
        projection: str = "linear",  # "linear" | "identity" | "tied"
        context_forward: int = 0,
        context_backward: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Positional encodings
        if pos_encoding_type == "learned":
            self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
            self.register_buffer("pos_embedding_rg", torch.arange(max_seq_len))
            self.get_positional_encoding = lambda seq_len: self.pos_embedding(
                self.pos_embedding_rg[:seq_len]
            )
        elif pos_encoding_type == "sin":
            self.sinusoidal_pos_encoder = _SinusoidalPositionEmb(
                hidden_dim, max_seq_len
            )
            self.get_positional_encoding = self.sinusoidal_pos_encoder.forward
        elif pos_encoding_type == "none":
            self.register_buffer("pos_embedding", torch.zeros(max_seq_len, hidden_dim))
            self.get_positional_encoding = lambda seq_len: self.pos_embedding[
                :seq_len, :
            ]
        else:
            raise ValueError("pos_encoding_type must be 'learned', 'sin', or 'none'")

        # Input projection and output projection
        self.projection = projection

        if self.projection == "identity":
            # No projection needed if dimensions match
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()
            assert hidden_dim == input_dim, (
                f"Hidden dimension must match input dimension ({input_dim}) for identity projection."
            )
        elif self.projection == "linear":
            self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
            self.output_projection = nn.Linear(hidden_dim, input_dim)
        elif self.projection == "tied":
            self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        elif self.projection == "input_only":
            self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
            self.output_projection = nn.Identity()
            assert hidden_dim == input_dim, (
                f"Hidden dimension must match input dimension ({input_dim}) for identity projection."
            )
        elif self.projection == "output_only":
            self.input_projection = nn.Identity()
            self.output_projection = nn.Linear(hidden_dim, input_dim)
        else:
            raise ValueError("projection must be 'linear', 'identity', or 'tied'")

        if self.projection in ("identity", "tied", "input_only"):
            self.output_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.output_bias = None

        # Transformer encoder layers
        def create_encoder_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [create_encoder_layer() for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)  # Dropout for rate regularization

        self.context_forward = context_forward
        self.context_backward = context_backward
        self.src_mask = {}

        # self.init_weights()

    def init_weights(self):
        r"""
        Init hoping for better optimization.
        Sources:
        Transformers without Tears https://arxiv.org/pdf/1910.05895.pdf
        T-Fixup http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        initrange = 0.1
        if self.projection in ("linear", "tied", "input_only"):
            self.input_projection.weight.data.uniform_(-initrange, initrange)

        if self.projection in ("linear", "output_only"):
            self.output_projection.weight.data.uniform_(-initrange, initrange)

    def _get_or_generate_context_mask(self, src, do_convert=True, expose_ic=True):
        # Create a context mask so that only some tokens are visible to others.
        # In effect, this makes the t'th token see between t - context_backward and t + context_forward tokens.
        if self.context_forward < 0 and self.context_backward < 0:
            return None

        if str(src.device) in self.src_mask:
            # Cached
            return self.src_mask[str(src.device)]

        # Generate context mask
        size = src.size(1)  # T
        context_forward = self.context_forward
        if self.context_forward < 0:
            context_forward = size
        mask = (
            torch.triu(
                torch.ones(size, size, device=src.device), diagonal=-context_forward
            )
            == 1
        ).transpose(0, 1)
        if self.context_backward > 0:
            back_mask = (
                torch.triu(
                    torch.ones(size, size, device=src.device),
                    diagonal=-self.context_backward,
                )
                == 1
            )
            mask = mask & back_mask

        # mask = mask.float()
        # if do_convert:
        #    mask = binary_mask_to_attn_mask(mask)
        self.src_mask[str(src.device)] = mask
        return self.src_mask[str(src.device)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer autoencoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, feature_dim = x.shape

        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input dimensions, got {feature_dim}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}."
            )

        # Project input and add positional encoding
        if self.projection not in ("identity", "output_only"):
            x = self.dropout(x)
        x = math.sqrt(self.hidden_dim) * self.input_projection(
            x
        ) + self.get_positional_encoding(seq_len).unsqueeze(0)
        x = self.dropout(x)

        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)  # Final layer normalization
        x = self.dropout(x)

        # Project back to input dimension
        if self.projection == "tied":
            # Use transposed weight matrix for tied weights (autoencoder style)
            x = F.linear(x, self.input_projection.weight.T, self.output_bias)
        elif self.output_bias is None:
            x = self.output_projection(x)
        else:
            x = x + self.output_bias

        return x

    def forward_until_layer(self, x, layer_num):
        batch_size, seq_len, feature_dim = x.shape

        x = self.dropout(x)
        x = math.sqrt(self.hidden_dim) * self.input_projection(
            x
        ) + self.get_positional_encoding(seq_len).unsqueeze(0)
        x = self.dropout(x)

        # Pass through transformer encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i == layer_num:
                break
        x = self.norm(x)  # Final layer normalization
        return x


def instantiate_autoencoder(args, n_neurons, trial_len):
    net = TransformerAutoencoder(
        input_dim=n_neurons,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=trial_len,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding,
        projection=args.projection,
        context_forward=args.context_forward,
        context_backward=args.context_backward,
    )
    return net


if __name__ == "__main__":
    # Example usage
    model = TransformerAutoencoder(
        input_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=1000,
        pos_encoding_type="sin",
        projection="linear",
        context_backward=-1,
        context_forward=0,
    )
    mask = model._get_or_generate_context_mask(
        torch.randn(1, 12, 1), do_convert=True, expose_ic=False
    )

    print(mask)
