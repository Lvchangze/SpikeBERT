import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import neuron
import numpy as np

# tau = 10.0 # beta = 1 - 1/tau
backend = "torch"
detach_reset=True
# common_thr = 1.0
# attn_thr = common_thr / 4

class spiking_self_attention(nn.Module):
    def __init__(self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_ln = nn.LayerNorm(dim)
        self.q_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.k_m = nn.Linear(dim, dim)
        self.k_ln = nn.LayerNorm(dim)
        self.k_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.v_m = nn.Linear(dim, dim)
        self.v_ln = nn.LayerNorm(dim)
        self.v_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.attn_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr/2, backend=backend)

        self.last_m = nn.Linear(dim, dim)
        self.last_ln = nn.LayerNorm(dim)
        self.last_lif = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

    def forward(self, x):# B T L D
        x = x.transpose(0, 1) # T B L D

        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1) # TB L D

        q_m_out = self.q_m(x_for_qkv) # TB L D
        q_m_out = self.q_ln(q_m_out).reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_ln(k_m_out).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_ln(v_m_out).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)
        x = (attn @ v) * self.qk_scale  # x_shape: T * B * heads * L * //heads
        # print(x.shape)

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        # print(x.shape)
        x = self.attn_lif(x)
        
        x = x.flatten(0, 1)
        # print(x.shape)
        x = self.last_m(x)
        x = self.last_ln(x)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())

        x = x.transpose(0, 1) # B T L D
        return x


class mlp(nn.Module):
    def __init__(self, length, tau, common_thr, in_features, hidden_features=None, out_features=None, ):
        super().__init__()
        # self.length = length
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.lif1 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.lif2 = neuron.LIFNode(tau=tau, step_mode='m', detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=common_thr, backend=backend)

    def forward(self, x):
        # B T L D
        x = x.transpose(0, 1) # T B L D
        T, B, L, D = x.shape
        x = x.flatten(0, 1)
        x = self.lif1(self.ln1(self.fc1(x)).reshape(T, B, L, D).contiguous())
        x = x.flatten(0, 1)
        x = self.lif2(self.ln2(self.fc2(x)).reshape(T, B, L, D).contiguous())
        x = x.transpose(0, 1) # B T L D
        return x


class block(nn.Module):
    def __init__(self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.125):
        super().__init__()
        self.attn = spiking_self_attention(length=length, tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = mlp(length=length, tau=tau, common_thr=common_thr, in_features=dim, hidden_features=dim*4, out_features=dim)

    def forward(self, x):
        # B T L D
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class transform(nn.Module):
    def __init__(self, dim, length):
        super(transform, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x


class new_spikformer(nn.Module):
    def __init__(self, depths, length, tau, common_thr, dim, T, vocab_size = 28996, num_classes=2, heads=8, qkv_bias=False, qk_scale=0.125, mode="train"):
        super().__init__()
        self.mode = mode
        self.atan = surrogate.ATan()
        self.T = T
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([block(
            length=length, tau=tau, common_thr=common_thr, dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale
        ) for _ in range(depths)])
        self.last_ln = nn.LayerNorm(dim)

        self.transforms = nn.ModuleList([
            transform(dim, length) for _ in range(depths)
        ])
        if mode != "pre_distill":
            self.classifier = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        # B L D
        x = self.emb(x)
        # print(x.shape)
        x = x.repeat(tuple([self.T] + torch.ones(len(x.size()), dtype=int).tolist())) # T B L D
        x = x.transpose(0, 1) # B T L D
        x = self.atan(x)
        representations = []
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B T L D
            representations.append(self.transforms[i](x.mean(1))) # B * L * D
            # last step
            # representations.append(self.transforms[i](x[:,-1,:,:])) # B * L * D
        # B T L D
        x = self.last_ln(x)
        # B T L D
        x = x.mean(2)
        if self.mode != "pre_distill":
            x = self.classifier(x)
        # x: B T D
        return representations, x
