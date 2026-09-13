'''主要是对距离进行角度和径向的编码'''
import torch
import torch.nn as nn
import sympy as sym

from dimenet_utils import (bessel_basis,real_sph_harm)'''分别负责径向和角度的计算'''
class Envelope(nn.Module):
'''对ra这个限制进行平滑处理，如果不做的话由于我们力的预测是基于对能量的微分进行的，不连续的性质可能会使预测性能大大降低，同时也可能影响预测准确度'''
'''平滑处理的实现是通过和径向函数相乘，其实就相当于在0<r<cutoff的距离内给距离加上一个权重，在距离外让径向函数值为0'''
def __init__(self, exponent=5):
        '''exponent越大计算复杂度越大，因此这里的选择应该是计算复杂度和平滑效果平衡的结果✨'''
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -(self.p * (self.p + 1)) / 2
    def forward(self, x):
        p = self.p
        a = self.a
        b = self.b
        c = self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        envelope = (
            1.0 / x
            + a * x_pow_p0
            + b * x_pow_p1
            + c * x_pow_p2
        )
        envelope = envelope * (x < 1.0).to(x.dtype)
        return envelope
'''公式具体是根据在cutoff处函数值，一阶，二阶导数为零推出来的'''
class BesselBasisLayer(nn.Module):
    def __init__(self,num_radial=6,cutoff=5.0,envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(
            exponent=envelope_exponent
        )
        self.freq = nn.Parameter(torch.empty(num_radial))'''初始化描述径向距离的频率，强调这个是可学习的'''
        self.reset_parameters()
    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(
                1,
                self.freq.numel() + 1,
                out=self.freq
            ).mul_(torch.pi)
        self.freq.requires_grad_()
    def forward(self, dist):
        dist = dist / self.cutoff
        envelope = self.envelope(dist)
        sinusoid = (
            self.freq * dist.unsqueeze(-1)
        ).sin()
        return envelope.unsqueeze(-1) * sinusoid
class SphericalBasisLayer(nn.Module):
    def __init__(self,num_spherical=7,num_radial=6,cutoff=5.0,envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(exponent=envelope_exponent)
        bessel_forms = bessel_basis(num_spherical,num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []
        x, theta = sym.symbols("x theta")
        modules = {"sin": torch.sin,"cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                self.sph_funcs.append(lambda x: torch.ones_like(x))
            else:
                sph = sym.lambdify([theta],sph_harm_forms[i][0],modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x],bessel_forms[i][j],modules)
                self.bessel_funcs.append(bessel)
    def forward(self,dist,angle,idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist)
                for f in self.bessel_funcs
            ],
            dim=1
        )
        rbf = (self.envelope(dist).unsqueeze(-1)* rbf)
        cbf = torch.stack(
            [
                f(angle)
                for f in self.sph_funcs
            ],
            dim=1
        )
        n = self.num_spherical
        k = self.num_radial
        out = (rbf[idx_kj].view(-1, n, k)*cbf.view(-1, n, 1))
        out = out.view(-1,n * k)

        return out
