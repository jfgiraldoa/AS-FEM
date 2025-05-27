#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 20:22:22 2025

@author: Juan Giraldo
"""

from dolfin import FacetNormal, CellDiameter, derivative, Constant, grad
from dg_forms import DGForms

class MatrixAssembler:
    """
    Assemble system matrices and RHS forms for ASFEM, pure DG, and mixed systems.

    :param argsDic: dict of parameters, including:
        'DOFset', 'mesh', 'f', 'automaticDIF', 'automaticDIF_DG',
        'Nonlinear', 'Solver'
    """
    def __init__(self, argsDic):
        self.argsDic = argsDic

    def system_matrices_asfem(self, e_k, u_k, e, u, w, v):
        """
        Build ASFEM bilinear/trilinear forms and RHS:
        returns A, F
        """
        Aargs = self.argsDic
        mesh = Aargs['mesh']
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)
        dg = DGForms(n,h,Aargs)

        # RHS form
        F = dg.rhs(u_k, w, Aargs['f'])

        # automatic differentiation path
        if Aargs['automaticDIF'] and Aargs['Nonlinear']:
            R1 = dg.gh(e_k, w, grad(e_k), grad(w), Constant(1)) + dg.bh(u_k, w) - F
            R2 = dg.bht(u_k, v)
            Gh  = derivative(R1, e_k, e)
            Bh  = derivative(R1, u_k, u)
            Bht = derivative(R2, u_k, e)
        else:
            Gh  = dg.gh(e, w, grad(e), grad(w),Constant(1))
            Bh  = dg.bh_prima(u_k, u, w)
            Bht = dg.bh_prima(u_k, v, e)

        # nonlinear correction
        if Aargs['Nonlinear']:
            R1     = dg.gh(e_k, w, grad(e_k), grad(w), Constant(1)) + dg.bh(u_k, w) - F
            R2     = dg.bht(u_k, v)
            Gh     = derivative(R1, e_k, e)
            Bh     = derivative(R1, u_k, u)
            Bhk    = dg.bh(u_k, w)
            Ghk    = dg.gh(e_k, w, grad(e_k), grad(w), Constant(1))
            if Aargs['automaticDIF']:
                Bhprimak = derivative(R2, u_k, e_k)
            else:
                Bhprimak = dg.bh_prima(u_k, v, e_k)
            F      += - Bhk - Bhprimak - Ghk

        # assemble global operator
        if Aargs['Solver'] == 'Iterative':
            A = [Gh, Bh, Bht]
        else:
            A = Gh + Bh + Bht
        return A, F

    def system_matrices_dg(self, u_k, u, w):
        """
        Build pure DG bilinear form and RHS: returns Bh, F
        """
        Aargs = self.argsDic
        mesh = Aargs['mesh']
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)
        dg = DGForms(n,h,Aargs)

        F = dg.rhs(u_k, w, Aargs['f'])
        if Aargs['automaticDIF_DG'] and Aargs['Nonlinear']:
            R1 = dg.bh(u_k, w) - F
            Bh = derivative(R1, u_k, u)
        else:
            Bh = dg.bh_prima(u_k, u, w)

        if Aargs['Nonlinear']:
            Bhk = dg.bh(u_k, w)
            F  += -Bhk
        return Bh, F

    def system_matrices_ms(self, u_k, e_k_fin, ud, wd):
        """
        Build mixed-scale system matrices: returns Bhd, Ghd, bhrhs
        """
        Aargs = self.argsDic
        mesh = Aargs['mesh']
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)
        dg = DGForms(n,h,Aargs)

        Ghd   = dg.gh(e_k_fin, wd, grad(e_k_fin), grad(wd), Constant(1))
        Bhd   = dg.bh_prima(u_k, ud, wd)
        bhrhs = dg.bh_prima(u_k, e_k_fin, wd)
        return Bhd, Ghd, bhrhs