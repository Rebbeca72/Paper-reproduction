import math
import numpy as np
import sympy as sym
import torch
from scipy.special import spherical_jn
from scipy.optimize import brentq
'''Dime的一些数学工具'''
def glorot_orthogonal(tensor,scale=2.0):'''对权重进行初始化，防止目标值经过权重以后越来越小'''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported.")
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = torch.empty(rows,cols,device=tensor.device,dtype=tensor.dtype)
    torch.nn.init.orthogonal_(flattened)
    norm = flattened.norm()
    target_norm = math.sqrt(scale * rows * cols / (rows + cols))
    flattened.mul_(target_norm / norm)
    with torch.no_grad():
        tensor.copy_(flattened.view_as(tensor))'''直接改变原来权重的值，而不是指向另外一个值'''
    return tensor
def Jn(x,n):
    return spherical_jn(n,x)
def Jn_zeros(n,k):
    zeros = np.zeros((n,k),dtype="float32")
    zeros[0] = (np.arange(1,k + 1)* np.pi)
    points = (np.arange(1,k + n)* np.pi)
    roots = np.zeros(k + n - 1,dtype="float32")
    for i in range(1,n):
        for j in range(k + n - 1 - i):
            root = brentq(Jn,points[j],points[j + 1],args=(i,))
            roots[j] = root
        points = roots.copy()
        zeros[i, :k] = roots[:k]
    return zeros
def spherical_bessel_formulas(n):
    x = sym.symbols("x")
    j = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1,n):
        b = sym.diff(a,x) / x
        j.append(sym.simplify(b * (-x) ** i))
        a = sym.simplify(b)
    return j
def bessel_basis(num_spherical,num_radial):
    zeros = Jn_zeros(num_spherical,num_radial)
    normalizer = []
    for order in range(num_spherical):
        normalizer_tmp = []
        for i in range(num_radial):
            normalizer_tmp.append(0.5* Jn(zeros[order, i],order + 1) ** 2)
        normalizer_tmp = (1/ np.array(normalizer_tmp) ** 0.5)
        normalizer.append(normalizer_tmp)
    formulas = spherical_bessel_formulas(num_spherical)
    x = sym.symbols("x")
    basis = []
    for order in range(num_spherical):
        radial_functions = []
        for i in range(num_radial):
            radial_function = sym.simplify(normalizer[order][i]*formulas[order].subs(x,zeros[order, i] * x))
            radial_functions.append(radial_function)
        basis.append(radial_functions)
    return basis
def sph_harm_prefactor(l,m):
    return math.sqrt(((2 * l + 1)* math.factorial(l - abs(m)))/(4* np.pi* math.factorial(l + abs(m))))
def associated_legendre_polynomials(L,zero_m_only=True):
    z = sym.symbols("z")
    P_l_m = [[0] * (l + 1)
        for l in range(L)
    ]
    P_l_m[0][0] = sym.Integer(1)
    if L == 1:
        return P_l_m
    P_l_m[1][0] = z
    if not zero_m_only:
        P_l_m[1][1] = -sym.sqrt(1 - z ** 2)
    for l in range(2,L):
        P_l_m[l][0] = sym.simplify(((2 * l - 1)* z* P_l_m[l - 1][0]-(l - 1)* P_l_m[l - 2][0])/ l)
        if not zero_m_only:
            P_l_m[l][l] = sym.simplify(-(2 * l - 1)* sym.sqrt(1 - z ** 2)* P_l_m[l - 1][l - 1])
            P_l_m[l][l - 1] = sym.simplify((2 * l - 1)* z* P_l_m[l - 1][l - 1])
            for m in range(0,l - 1):
                P_l_m[l][m] = sym.simplify(((2 * l - 1)* z* P_l_m[l - 1][m]-(l + m - 1)* P_l_m[l - 2][m])/(l - m))
    return P_l_m
def real_sph_harm(L,zero_m_only=True):
    z = sym.symbols("z")
    P_l_m = associated_legendre_polynomials(L,zero_m_only)
    if zero_m_only:
        Y_l_m = [[0]
            for _ in range(L)
        ]
    else:
        Y_l_m = [
            [0] * (2 * l + 1)
            for l in range(L)
        ]
    theta = sym.symbols("theta")
    for l in range(L):
        for m in range(len(P_l_m[l])):
            if not isinstance(P_l_m[l][m],int):
                P_l_m[l][m] = sym.simplify(P_l_m[l][m].subs(z,sym.cos(theta)))
    for l in range(L):
        Y_l_m[l][0] = sym.simplify(sph_harm_prefactor(l,0)*P_l_m[l][0])
    if not zero_m_only:
        phi = sym.symbols("phi")
        for l in range(1,L):
            for m in range(1,l + 1):
                Y_l_m[l][m] = sym.simplify(math.sqrt(2)* (-1) ** m* sph_harm_prefactor(l,m)* P_l_m[l][m]* sym.cos(m * phi))
            for m in range(1,l + 1):
                Y_l_m[l][-m] = sym.simplify(math.sqrt(2)* (-1) ** m* sph_harm_prefactor(l,-m)* P_l_m[l][m]* sym.sin(m * phi))
    return Y_l_m
