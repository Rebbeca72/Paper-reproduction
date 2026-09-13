import math

import numpy as np
import sympy as sym
import torch

from scipy.special import spherical_jn
from scipy.optimize import brentq


def glorot_orthogonal(
    tensor,
    scale=2.0
):
    if tensor.ndimension() < 2:
        raise ValueError(
            "Only tensors with 2 or more dimensions are supported."
        )

    rows = tensor.size(0)
    cols = tensor.numel() // rows

    flattened = torch.empty(
        rows,
        cols,
        device=tensor.device,
        dtype=tensor.dtype
    )

    torch.nn.init.orthogonal_(
        flattened
    )

    norm = flattened.norm()

    target_norm = math.sqrt(
        scale * rows * cols / (rows + cols)
    )

    flattened.mul_(
        target_norm / norm
    )

    with torch.no_grad():
        tensor.copy_(
            flattened.view_as(tensor)
        )

    return tensor


def Jn(
    x,
    n
):
    """
    Spherical Bessel function j_n(x).
    """

    return spherical_jn(
        n,
        x
    )


def Jn_zeros(
    n,
    k
):
    """
    Compute the first k zeros of spherical
    Bessel functions j_l(x), for l = 0,...,n-1.
    """

    zeros = np.zeros(
        (
            n,
            k
        ),
        dtype="float32"
    )

    # --------------------------------------------------
    # l = 0
    #
    # j_0(x) = sin(x) / x
    #
    # Its zeros are:
    #
    # pi, 2pi, 3pi, ...
    # --------------------------------------------------

    zeros[0] = (
        np.arange(
            1,
            k + 1
        )
        * np.pi
    )

    # --------------------------------------------------
    # l > 0
    #
    # Find zeros numerically.
    # --------------------------------------------------

    points = (
        np.arange(
            1,
            k + n
        )
        * np.pi
    )

    roots = np.zeros(
        k + n - 1,
        dtype="float32"
    )

    for i in range(
        1,
        n
    ):

        for j in range(
            k + n - 1 - i
        ):

            root = brentq(
                Jn,
                points[j],
                points[j + 1],
                args=(i,)
            )

            roots[j] = root

        points = roots.copy()

        zeros[i, :k] = roots[:k]

    return zeros


def spherical_bessel_formulas(
    n
):
    """
    Generate symbolic formulas for spherical
    Bessel functions:

        j_0(x)
        j_1(x)
        j_2(x)
        ...

    """

    x = sym.symbols("x")

    # j_0(x)
    j = [
        sym.sin(x) / x
    ]

    a = sym.sin(x) / x

    for i in range(
        1,
        n
    ):

        b = sym.diff(
            a,
            x
        ) / x

        j.append(
            sym.simplify(
                b * (-x) ** i
            )
        )

        a = sym.simplify(
            b
        )

    return j


def bessel_basis(
    num_spherical,
    num_radial
):
    """
    Construct normalized spherical Bessel basis.

    Returns:
        basis[l][n]

    where:

        l = spherical order
        n = radial basis index
    """

    zeros = Jn_zeros(
        num_spherical,
        num_radial
    )

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------

    normalizer = []

    for order in range(
        num_spherical
    ):

        normalizer_tmp = []

        for i in range(
            num_radial
        ):

            normalizer_tmp.append(
                0.5
                * Jn(
                    zeros[order, i],
                    order + 1
                ) ** 2
            )

        normalizer_tmp = (
            1
            / np.array(
                normalizer_tmp
            ) ** 0.5
        )

        normalizer.append(
            normalizer_tmp
        )

    # --------------------------------------------------
    # Symbolic spherical Bessel formulas
    # --------------------------------------------------

    formulas = spherical_bessel_formulas(
        num_spherical
    )

    x = sym.symbols("x")

    basis = []

    for order in range(
        num_spherical
    ):

        radial_functions = []

        for i in range(
            num_radial
        ):

            radial_function = sym.simplify(

                normalizer[order][i]
                *
                formulas[order].subs(
                    x,
                    zeros[order, i] * x
                )

            )

            radial_functions.append(
                radial_function
            )

        basis.append(
            radial_functions
        )

    return basis


def sph_harm_prefactor(
    l,
    m
):
    """
    Normalization factor for real spherical harmonics.
    """

    return math.sqrt(
        (
            (2 * l + 1)
            * math.factorial(
                l - abs(m)
            )
        )
        /
        (
            4
            * np.pi
            * math.factorial(
                l + abs(m)
            )
        )
    )


def associated_legendre_polynomials(
    L,
    zero_m_only=True
):
    """
    Generate associated Legendre polynomials
    P_l^m(z).

    Only m=0 is needed by the original
    DimeNet SphericalBasisLayer.
    """

    z = sym.symbols("z")

    P_l_m = [
        [0] * (l + 1)
        for l in range(L)
    ]

    # P_0^0
    P_l_m[0][0] = sym.Integer(1)

    if L == 1:
        return P_l_m

    # P_1^0
    P_l_m[1][0] = z

    if not zero_m_only:
        P_l_m[1][1] = -sym.sqrt(
            1 - z ** 2
        )

    for l in range(
        2,
        L
    ):

        # P_l^0
        P_l_m[l][0] = sym.simplify(
            (
                (2 * l - 1)
                * z
                * P_l_m[l - 1][0]
                -
                (l - 1)
                * P_l_m[l - 2][0]
            )
            / l
        )

        if not zero_m_only:

            # P_l^l
            P_l_m[l][l] = sym.simplify(
                -(2 * l - 1)
                * sym.sqrt(
                    1 - z ** 2
                )
                * P_l_m[l - 1][l - 1]
            )

            # P_l^(l-1)
            P_l_m[l][l - 1] = sym.simplify(
                (2 * l - 1)
                * z
                * P_l_m[l - 1][l - 1]
            )

            for m in range(
                0,
                l - 1
            ):

                P_l_m[l][m] = sym.simplify(
                    (
                        (2 * l - 1)
                        * z
                        * P_l_m[l - 1][m]
                        -
                        (l + m - 1)
                        * P_l_m[l - 2][m]
                    )
                    /
                    (l - m)
                )

    return P_l_m


def real_sph_harm(
    L,
    zero_m_only=True
):
    """
    Generate real spherical harmonics.

    Original DimeNet uses only m=0.
    """

    z = sym.symbols("z")

    P_l_m = associated_legendre_polynomials(
        L,
        zero_m_only
    )

    if zero_m_only:

        Y_l_m = [
            [0]
            for _ in range(L)
        ]

    else:

        Y_l_m = [
            [0] * (2 * l + 1)
            for l in range(L)
        ]

    theta = sym.symbols(
        "theta"
    )

    # --------------------------------------------------
    # Convert z -> cos(theta)
    # --------------------------------------------------

    for l in range(L):

        for m in range(
            len(P_l_m[l])
        ):

            if not isinstance(
                P_l_m[l][m],
                int
            ):

                P_l_m[l][m] = sym.simplify(
                    P_l_m[l][m].subs(
                        z,
                        sym.cos(theta)
                    )
                )

    # --------------------------------------------------
    # m = 0
    # --------------------------------------------------

    for l in range(L):

        Y_l_m[l][0] = sym.simplify(
            sph_harm_prefactor(
                l,
                0
            )
            *
            P_l_m[l][0]
        )

    # --------------------------------------------------
    # m != 0
    #
    # Not used by original DimeNet,
    # but retained for completeness.
    # --------------------------------------------------

    if not zero_m_only:

        phi = sym.symbols(
            "phi"
        )

        for l in range(
            1,
            L
        ):

            # m > 0
            for m in range(
                1,
                l + 1
            ):

                Y_l_m[l][m] = sym.simplify(

                    math.sqrt(2)
                    * (-1) ** m
                    * sph_harm_prefactor(
                        l,
                        m
                    )
                    * P_l_m[l][m]
                    * sym.cos(
                        m * phi
                    )

                )

            # m < 0
            for m in range(
                1,
                l + 1
            ):

                Y_l_m[l][-m] = sym.simplify(

                    math.sqrt(2)
                    * (-1) ** m
                    * sph_harm_prefactor(
                        l,
                        -m
                    )
                    * P_l_m[l][m]
                    * sym.sin(
                        m * phi
                    )

                )

    return Y_l_m