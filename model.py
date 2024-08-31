import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # n.o of head for queries
    n_kv_heads: Optional[int] = None  # n.o of heads for k and v
    vocab_size: int = -1  # will be set when tokenizer is loaded
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # KV cache

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    # dim of embedding should be even as per paper
    assert head_dim % 2 == 0, "Dimension must be div by 2"

    # formula theta_i=10000^(-2(i-1)/dim) for i =[1,2,... dim/2]
    # shape:(head_dim/2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Position params (m)
    # shape: (seq_len)

    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position
    freqs = torch.outer(m, theta).float()
    # compute complex numbers in polar form c= R*exp(i*m*theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size is not set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)  # (B,seq_len) -> (b,seq_len,dim)

        freqs_complex = self.freqs_complex[
            start_pos : start_pos + seq_len
        ]  # get (m,theta) corresponding to the positions

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)

        output = self.output(h).float()
        return output


# Llama2 embedding -> 4096
