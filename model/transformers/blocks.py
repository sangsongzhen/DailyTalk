import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().transpose(1, 2)

        return x


# class FiLM(nn.Module):
#     """
#     A Feature-wise Linear Modulation Layer
#     """
#     def __init__(self):
#         super(FiLM, self).__init__()
#         self.s_gamma = nn.Parameter(torch.ones(1), requires_grad=True)
#         self.s_beta = nn.Parameter(torch.ones(1), requires_grad=True)

#     def forward(self, x, gammas, betas):
#         """
#         x: [B, T, H]
#         gammas: [B, 1, H] or [B, H]
#         betas:  [B, 1, H] or [B, H]
#         """
#         if gammas.dim() == 2:
#             gammas = gammas.unsqueeze(1)
#         if betas.dim() == 2:
#             betas = betas.unsqueeze(1)

#         gammas = self.s_gamma * gammas.expand_as(x)
#         betas = self.s_beta * betas.expand_as(x)
#         return (1.0 + gammas) * x + betas

# Advanced FiLM
class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer with bounded gamma/beta
    """
    def __init__(self, min_gamma=0.5, max_gamma=1.5, max_beta=1.0):
        super(FiLM, self).__init__()
        self.s_gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.s_beta = nn.Parameter(torch.ones(1), requires_grad=True)

        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.max_beta = max_beta

    def forward(self, x, gammas, betas):
        """
        x: [B, T, H]
        gammas: [B, 1, H] or [B, H]
        betas:  [B, 1, H] or [B, H]
        """
        if gammas.dim() == 2:
            gammas = gammas.unsqueeze(1)  # [B, 1, H]
        if betas.dim() == 2:
            betas = betas.unsqueeze(1)    # [B, 1, H]

        # Scale modulation vectors
        gammas = self.s_gamma * gammas.expand_as(x)
        betas = self.s_beta * betas.expand_as(x)

        # Clamp to avoid extreme modulation
        gammas = torch.clamp(gammas, self.min_gamma - 1.0, self.max_gamma - 1.0)  # 调整 (1 + γ)
        betas = torch.clamp(betas, -self.max_beta, self.max_beta)

        return (1.0 + gammas) * x + betas
