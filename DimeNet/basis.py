import torch
import torch.nn as nn
import sympy as sym

from dimenet_utils import (
    bessel_basis,
    real_sph_harm
)


# ============================================================
# Envelope
# ============================================================

class Envelope(nn.Module):

    def __init__(self, exponent=5):

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


# ============================================================
# Bessel Basis Layer
# ============================================================

class BesselBasisLayer(nn.Module):

    def __init__(
        self,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5
    ):

        super().__init__()

        self.cutoff = cutoff

        self.envelope = Envelope(
            exponent=envelope_exponent
        )

        self.freq = nn.Parameter(
            torch.empty(num_radial)
        )

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

        # ----------------------------------------------------
        # dist:
        # [E]
        #
        # E = number of edges
        # ----------------------------------------------------

        dist = dist / self.cutoff

        # ----------------------------------------------------
        # Envelope
        #
        # [E]
        # ----------------------------------------------------

        envelope = self.envelope(dist)

        # ----------------------------------------------------
        # dist.unsqueeze(-1):
        #
        # [E]
        #     ↓
        # [E, 1]
        #
        # freq:
        #
        # [num_radial]
        # = [6]
        #
        # 最终：
        #
        # [E, 1] * [6]
        #     ↓
        # [E, 6]
        # ----------------------------------------------------

        sinusoid = (
            self.freq * dist.unsqueeze(-1)
        ).sin()

        # ----------------------------------------------------
        # [E, 1] * [E, 6]
        #        ↓
        #      [E, 6]
        # ----------------------------------------------------

        return envelope.unsqueeze(-1) * sinusoid


# ============================================================
# Spherical Basis Layer
# ============================================================

class SphericalBasisLayer(nn.Module):

    def __init__(
        self,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5
    ):

        super().__init__()

        self.num_spherical = num_spherical

        self.num_radial = num_radial

        self.cutoff = cutoff

        self.envelope = Envelope(
            exponent=envelope_exponent
        )

        # ----------------------------------------------------
        # spherical Bessel functions
        # ----------------------------------------------------

        bessel_forms = bessel_basis(
            num_spherical,
            num_radial
        )

        # ----------------------------------------------------
        # real spherical harmonics
        # ----------------------------------------------------

        sph_harm_forms = real_sph_harm(
            num_spherical
        )

        self.sph_funcs = []

        self.bessel_funcs = []

        x, theta = sym.symbols(
            "x theta"
        )

        modules = {
            "sin": torch.sin,
            "cos": torch.cos
        }

        # ====================================================
        # Generate spherical functions
        # ====================================================

        for i in range(num_spherical):

            # ------------------------------------------------
            # l = 0
            #
            # Y_0^0 = constant
            # ------------------------------------------------

            if i == 0:

                self.sph_funcs.append(
                    lambda x: torch.ones_like(x)
                )

            else:

                sph = sym.lambdify(
                    [theta],
                    sph_harm_forms[i][0],
                    modules
                )

                self.sph_funcs.append(
                    sph
                )

            # ------------------------------------------------
            # Generate radial Bessel functions
            # ------------------------------------------------

            for j in range(num_radial):

                bessel = sym.lambdify(
                    [x],
                    bessel_forms[i][j],
                    modules
                )

                self.bessel_funcs.append(
                    bessel
                )

    def forward(
        self,
        dist,
        angle,
        idx_kj
    ):

        # ====================================================
        # 1. Normalize distance
        # ====================================================

        dist = dist / self.cutoff

        # ====================================================
        # 2. Bessel radial basis
        # ====================================================

        rbf = torch.stack(
            [
                f(dist)
                for f in self.bessel_funcs
            ],
            dim=1
        )

        # ----------------------------------------------------
        # rbf:
        #
        # [E, num_spherical * num_radial]
        #
        # [E, 42]
        # ----------------------------------------------------

        rbf = (
            self.envelope(dist)
            .unsqueeze(-1)
            * rbf
        )

        # ====================================================
        # 3. Spherical harmonic angular basis
        # ====================================================

        cbf = torch.stack(
            [
                f(angle)
                for f in self.sph_funcs
            ],
            dim=1
        )

        # ----------------------------------------------------
        # cbf:
        #
        # [T, num_spherical]
        #
        # [T, 7]
        # ----------------------------------------------------

        # ====================================================
        # 4. Combine radial + angular basis
        # ====================================================

        n = self.num_spherical

        k = self.num_radial

        # ----------------------------------------------------
        # rbf[idx_kj]
        #
        # idx_kj:
        # [T]
        #
        # rbf:
        # [E, n*k]
        #
        # 得到：
        # [T, n*k]
        #
        # reshape：
        # [T, n, k]
        # ----------------------------------------------------

        out = (
            rbf[idx_kj]
            .view(-1, n, k)
            *
            cbf.view(-1, n, 1)
        )

        # ----------------------------------------------------
        # [T, n, k]
        #     ↓
        # [T, n*k]
        #
        # [T, 42]
        # ----------------------------------------------------

        out = out.view(
            -1,
            n * k
        )

        return out