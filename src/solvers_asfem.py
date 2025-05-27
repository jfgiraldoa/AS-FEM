#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:31:26 2025

@author: Juan Giraldo
"""
from dolfin import *
from function_space_utils import FunctionSpaceUtils as fsu
from dg_forms import DGForms
import time
import scipy.sparse as sp
from sksparse.cholmod import cholesky
import scipy.sparse.linalg as spl
import numpy as np

class Solvers:
    def __init__(self, args_dic):
        """
        args_dic: dictionary with solver parameters, e.g.
          'Ptype', 'ilu_drop_tol', 'ilu_fill_factor',
          'Gtype','Tol_gram','Stype','Tol_schur','max_iter_cg',
          'Hh':[V_space, U_space, mesh], 'mesh', 'DOFset',
          'iterR','Esol','UsolCG','f', 'gN','g2N',
          'gR_alpha','gR_u'
        """
        self.args = args_dic
        self.call_sum = 0

    def solverfun(self, A, l, V, solver_type):
        """
        Master solve method dispatching to direct/iterative solvers.
        A: either UFL forms or tuple of forms (for Iterative)
        l: right-hand UFL form
        V: function space for direct solvers
        solver_type: one of 'Direct','DirectDG','mumps','mumpsDG','gmres','gmresDG','PETSCDG','Iterative'
        """
        if solver_type == 'Direct':
            delta = Function(V)
            solve(A == l, delta)
            e_k, u_k = delta.split(True)

        elif solver_type == 'DirectDG':
            u_k = Function(V)
            solve(A == l, u_k)
            e_k = []

        elif solver_type == 'mumps':
            delta = Function(V)
            solve(A == l, delta,
                  solver_parameters={"linear_solver": "mumps"},
                  form_compiler_parameters={"optimize": True})
            e_k, u_k = delta.split(True)

        elif solver_type == 'mumpsDG':
            u_k = Function(V)
            solve(A == l, u_k,
                  solver_parameters={"linear_solver": "mumps"},
                  form_compiler_parameters={"optimize": True})
            e_k = []

        elif solver_type == 'gmres':
            delta = Function(V)
            solve(A == l, delta,
                  solver_parameters={"linear_solver": "gmres", 'preconditioner': 'ilu'},
                  form_compiler_parameters={"optimize": True})
            e_k, u_k = delta.split(True)

        elif solver_type == 'gmresDG':
            u_k = Function(V)
            solve(A == l, u_k,
                  solver_parameters={"linear_solver": "gmres", 'preconditioner': 'ilu'},
                  form_compiler_parameters={"optimize": True})
            e_k = []

        elif solver_type == 'PETSCDG':
            u_k = Function(V)
            solve(A == l, u_k,
                  solver_parameters={"linear_solver": "petsc", 'preconditioner': 'petsc_amg'})
            e_k = []

        elif solver_type == 'Iterative':
            ge, be, be_t = A
            e_k, u_k = self.iterative_solver(ge, be, be_t, l)

        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        return e_k, u_k

    def _preconditioner(self, G, NdofsV):
        """Build preconditioner operator for Gram matrix."""
        P_op = None
        p = self.args
        if p['Ptype'] == 'CHOLESKY':
            mode32 = NdofsV <= 3e6
            start = time.time()
            P = cholesky(G, mode="auto", use_long=mode32)
            P_op = spl.LinearOperator((NdofsV, NdofsV), lambda x: P.solve_A(x))
            print(f"Elapsed Cholesky: {time.time()-start:.3e}s")
        elif p['Ptype'] == 'ILU':
            start = time.time()
            Pmat = spl.spilu(G, drop_tol=p['ilu_drop_tol'], fill_factor=p['ilu_fill_factor'])
            P_op = spl.LinearOperator((NdofsV, NdofsV), lambda x: Pmat.solve(x))
            print(f"Elapsed ILU: {time.time()-start:.3e}s")
        return P_op

    def _cg_solver(self, A, b, x0, P, Niter, tol):
        """Run CG, counting calls."""
        self.call_sum += 1
        val, _ = spl.cg(A, b, M=P, x0=x0, maxiter=Niter, tol=tol)
        return val

    def _schur(self, B, Ginv, B_t, b):
        """Apply Schur complement: B_t * Ginv(B * x)."""
        return B_t.dot(Ginv(B.dot(b)))

    def _schur_solver(self, A, b, tol_s, prec, maxiter):
        """Solve Schur system by CG with callback."""
        iters = 0
        def cb(xk): nonlocal iters; iters += 1
        du, _ = spl.cg(A, b, tol=tol_s, M=prec, maxiter=maxiter, callback=cb)
        print(f"Schur CG iters = {iters}")
        return du, iters
    
    
    def iterative_solver(self, ge, be, be_t, le):
        """Full block-iterative solution using Schur and DGForms."""
        p = self.args
        V, U = p['Hh'][0], p['Hh'][1]
        e, u, v, w = fsu.get_TestTrial(p['Hh'][2], p)
        n = FacetNormal(p['mesh'])
        h = CellDiameter(p['mesh'])
        dg = DGForms(n, h, p)
    
        # assemble matrices
        t0 = time.time()
        Gm = as_backend_type(assemble(ge)).mat()
        G  = sp.csc_matrix(sp.csr_matrix(Gm.getValuesCSR()[::-1], shape=Gm.size))
        G.eliminate_zeros()
    
        Bm = as_backend_type(assemble(be)).mat()
        B  = sp.csc_matrix(sp.csr_matrix(Bm.getValuesCSR()[::-1], shape=Bm.size))
        B.eliminate_zeros()
    
        Bt_m = as_backend_type(assemble(be_t)).mat()
        B_t  = sp.csr_matrix(Bt_m.getValuesCSR()[::-1], shape=Bt_m.size)
        B_t.eliminate_zeros()
    
        L = assemble(le)
        print(f"Assembly time = {time.time()-t0:.3f}s")
    
        # preconditioner for Gram
        P_op = self._preconditioner(G, p['DOFset'][0])
        if p['Gtype'] == 'NONE':
            Ginv = lambda x: P_op(x)
        elif p['Gtype'] == 'CG':
            Ginv = lambda x: self._cg_solver(G, x, P_op(x),
                                             P_op, p['DOFset'][1], p['Tol_gram'])
    
        # Schur operator and preconditioner
        S_op = spl.LinearOperator((p['DOFset'][1],)*2,
                                  lambda x: B_t.dot(Ginv(B.dot(x))))
        if p['Stype'] == 'CHOLESKY':
            d = 1/G.diagonal()
            Sprec = cholesky(B_t*sp.diags(d)*B, mode="auto")
            Sch_inv = spl.LinearOperator((p['DOFset'][1],)*2,
                                         lambda x: Sprec.solve_A(x))
        else:
            Sch_inv = None
    
        # initial guesses
        e_prev, u_prev = Function(V), Function(U)
        deltae, deltau = Function(V), Function(U)
        if p['iterR'] == 0:
            res_e = np.array(L)
            res_u = np.zeros(p['DOFset'][1])
        else:
            e_prev.assign(p['Esol'])
            u_prev.assign(p['UsolCG'])
            lh1  = dg.rhs(u_prev, w, p['f'])
            bh1  = dg.bh(u_prev, w)
            gh1  = dg.gh(e_prev, w, grad(e_prev), grad(w), Constant(1))
            bht1 = dg.bht(v, e_prev)
            res_e = np.array(assemble(lh1 - bh1 - gh1))
            res_u = np.array(assemble(-bht1))
    
        # iterative Schur-CG loop
        tol_s, maxit = p['Tol_schur'], p['max_iter_cg']
        ave, itercg = 1, 0
        sum0 = np.sum(res_e**2) + np.sum(res_u**2)
        while itercg < maxit and ave > p['tol_ave']:
            itercg += 1
            du, nit = self._schur_solver(S_op,
                                         B_t.dot(Ginv(res_e)) - res_u,
                                         tol_s, Sch_inv, maxit)
            de = Ginv(res_e - B.dot(du))
            deltau.vector()[:] += du
            deltae.vector()[:] += de
            res_e += -G.dot(de) - B.dot(du)
            res_u += -B_t.dot(de)
            sumtot = np.sum(res_e**2) + np.sum(res_u**2)
            ave = sumtot/sum0
            print(f"iter {itercg} ave = {ave}")
    
        # final projection
        u_prevn = fsu.projection(u_prev, U, 'Vh', 0, p)
        e_prevn = fsu.projection(e_prev, V, 'Vh', P_op, p)
        u_new = Function(U)
        e_new = Function(V)
        u_new.vector()[:] = deltau.vector()[:] + u_prevn.vector()[:]
        e_new.vector()[:] = deltae.vector()[:] + e_prevn.vector()[:]
        return e_new, u_new
