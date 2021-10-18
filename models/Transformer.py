import torch.nn as nn
import torch
from torch.nn import Transformer

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings=None, embedding_dim=None, seq_length=None):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 1, 256)) # 8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings



class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim=256, max_length=1024):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x



class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()

        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Residual(nn.Module):
    def __init__(self, fn, weights=False):
        super().__init__()
        self.fn = fn
        self.weights = weights

    def forward(self, x):
        if self.weights:
            ouput, weights = self.fn(x)
            return ouput + x
        else:
            return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn, weights=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn
        self.weights = weights

    def forward(self, x):
        attended_features, attention_weights = self.fn(self.norm(x))

        if self.weights:
            return self.dropout(attended_features), attention_weights
        else:
            return self.dropout(attended_features)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            intermediate=False,
            weights=False
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                            weights=weights
                        ),
                        weights=weights
                    ),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers, return_intermediate=intermediate)

    def forward(self, x):
        return self.net(x)



def execute_hook(module, input, output, array):
    out, weight = output
    array.append(weight)

if __name__ == '__main__':
    t = TransformerModel(256, 4, 4, 512, intermediate=False, weights=True)
    print('# netRelighting parameters:', sum(param.numel() for param in t.parameters()))
    T = Transformer(256, 4, 4, dim_feedforward=512).encoder
    # print(T)
    print(t)

    # print(t.net[0].fn.fn)

    # use lists to store the outputs via up-values
    enc_attn_weights = []

    hooks = t.net[0].fn.fn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
    )

    hook1 = t.net[2].fn.fn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    )

    hooks2 = t.net[4].fn.fn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    )

    hooks3 = t.net[6].fn.fn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    )



    i = torch.randn(1, 64, 256)
    o = t(i)
    print(enc_attn_weights[0].shape)

    # print(o[1].shape)
