#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:45:03 2025

@author: Juan Giraldo
"""

from dolfin import *
import numpy as np
import scipy.sparse.linalg as spl
from dg_forms import DGForms
import scipy.sparse as sp


class FunctionSpaceUtils:
    @staticmethod
    def get_FunctionSpaces(mesh, params):
        """
        Build mixed and subspace function spaces.

        :param mesh:    DOLFIN Mesh
        :param params:  dict with keys 'TestType','Ptest','TrialType','Ptrial'
        :return: V1, V2, V
          V1 = DG space for e
          V2 = CG space for u
          V  = Mixed space V1 x V2
        """
        V1_ele = FiniteElement(params['TestType'], mesh.ufl_cell(), params['Ptest'])
        V2_ele = FiniteElement(params['TrialType'], mesh.ufl_cell(), params['Ptrial'])
        mixed  = MixedElement(V1_ele, V2_ele)
        V       = FunctionSpace(mesh, mixed)

        # collapse to get individual spaces
        V1 = V.sub(0).collapse()
        V2 = V.sub(1).collapse()
        return V1, V2, V

    @staticmethod
    def get_TestTrial(V, args_dic):
        """
        Return appropriate trial/test functions depending on solver type.

        :param V:          Mixed FunctionSpace
        :param args_dic:   dict with key 'Solver'
        :return: e, u, v, w
        """
        if args_dic['Solver'] == 'Iterative':
            V1 = V.sub(0).collapse()
            V2 = V.sub(1).collapse()
            e = TrialFunction(V1)
            u = TrialFunction(V2)
            w = TestFunction(V1)
            v = TestFunction(V2)
        else:
            e, u = TrialFunctions(V)
            w, v = TestFunctions(V)
        return e, u, v, w

    @staticmethod
    def get_spaceDofs(V):
        """
        Count DOFs in subspaces of a mixed space.

        :param V: Mixed FunctionSpace
        :return: Ndofs0, Ndofs1, total_dofs
        """
        dof0 = V.sub(0).dofmap().dofs()
        dof1 = V.sub(1).dofmap().dofs()
        Ndofs0 = np.size(dof0)
        Ndofs1 = np.size(dof1)
        total = Ndofs0 + Ndofs1
        print(f"dof0 = {Ndofs0}, dof1 = {Ndofs1}, total = {total}")
        return Ndofs0, Ndofs1, total

    @staticmethod
    def projection(u1, V, type, P, args_dic):
        """
        Project a function onto a given space and norm.

        :param u1:      source Function
        :param V:       target FunctionSpace
        :param type:    'L2', 'H1', or 'Vh'
        :param P:       preconditioner operator or None
        :param args_dic: arguments dictionary with measures and DGForms
        :return: proj   projected Function
        """
        proj = Function(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        uf = Function(V)
        uf.assign(u1)

        if type == 'L2':
            m = inner(u, v) * args_dic['dx']
            Me = assemble(m)
            M_mat = as_backend_type(Me).mat()
            del Me
            M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape=M_mat.size))
            rhs = assemble(inner(uf, v) * args_dic['dx'])
            BigM = spl.splu(M)
            proj.vector()[:] = BigM.solve(np.array(rhs))
            del rhs, M

        elif type == 'H1':
            m = inner(u, v) * args_dic['dx'] + inner(grad(u), grad(v)) * args_dic['dx']
            Me = assemble(m)
            M_mat = as_backend_type(Me).mat()
            del Me
            M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape=M_mat.size))
            rhs = assemble(inner(uf, v) * args_dic['dx'])
            rhs += assemble(inner(grad(uf), grad(v)) * args_dic['dx'])
            val, _ = spl.cg(M, rhs, tol=1e-7)
            proj.vector()[:] = val

        elif type == 'Vh':
            n = FacetNormal(args_dic['mesh'])
            h = CellDiameter(args_dic['mesh'])
            dg = DGForms(n, h, args_dic)

            Me = assemble(dg.gh(u, v, grad(u), grad(v), Constant(1.0)))
            M_mat = as_backend_type(Me).mat()
            M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape=M_mat.size))
            aa = dg.gh(uf, v, grad(uf), grad(v), Constant(1.0))
            rhs = assemble(aa)
            if P:
                val, _ = spl.cg(M, rhs, M=P, tol=1e-7)
            else:
                val, _ = spl.cg(M, rhs, tol=1e-7)
            proj.vector()[:] = val

        return proj
    
    @staticmethod
    def normresidual(e_k,u_k,w,v,ArgsDic):
        n   = FacetNormal(ArgsDic['mesh'])
        h   = CellDiameter(ArgsDic['mesh'])
        dg = DGForms(n,h,ArgsDic)
        Lh  = dg.rhs(u_k,w,ArgsDic['f'])
        Bh  = dg.bh(u_k,w)
        Bhp = dg.bh_prima(u_k,v,e_k)
        Gh  = dg.gh(e_k,w,grad(e_k),grad(w),Constant(1))

        return np.linalg.norm(np.array(assemble(Lh-Bh-Bhp-Gh)))

    @staticmethod
    def normresidualDG(u_k,w,ArgsDic):
        n   = FacetNormal(ArgsDic['mesh'])
        h   = CellDiameter(ArgsDic['mesh'])
        dg = DGForms(n,h,ArgsDic)
        Lh  = dg.rhs(u_k,w,ArgsDic['f'])
        Bh  = dg.bh(u_k,w)
        return np.linalg.norm(np.array(assemble(Lh-Bh)))
    
    @staticmethod
    def disc_proj_Uh(u1, U, dim):
        proj = Function(U)
        u = TrialFunction(U)
        v = TestFunction(U)
        dof_coords = U.tabulate_dof_coordinates().reshape(-1, dim)
        for j in range(len(dof_coords)):
            proj.vector()[j] = u1(dof_coords[j])
        return proj
    
    @staticmethod
    def projection_exa(u1,gradu1,V,type,P,ArgsDic):
         proj = Function(V)
         u = TrialFunction(V)
         v = TestFunction(V)
         uf = Function(V)
         uf.assign(u1)
         if type=='Vh':
             n = FacetNormal(ArgsDic['mesh'])
             h = CellDiameter(ArgsDic['mesh'])
             dg = DGForms(n,h,ArgsDic)
             Me = assemble(dg.gh(u,v,grad(u),grad(v),Constant(1.)))
             M_mat = as_backend_type(Me).mat()
             M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape = M_mat.size))
             rhs = assemble(dg.gh(u1,v,gradu1,grad(v),n,h,Constant(1.)))
             if P:  val,auuu = spl.cg(M,rhs,M=P,tol=1e-7)
             else:  val,auuu = spl.cg(M,rhs,tol=1e-7)
             proj.vector()[:] = val   
         elif type=='H1':
             m = inner(u,v)*ArgsDic['dx']+ inner(grad(u),grad(v))*dx
             Me = assemble(m)
             M_mat = as_backend_type(Me).mat()
             del Me
             M = sp.csc_matrix(sp.csr_matrix(M_mat.getValuesCSR()[::-1], shape = M_mat.size))
             rhs = assemble(inner(u1,v)*ArgsDic['dx']) + assemble(inner(gradu1,grad(v))*ArgsDic['dx']) 
             val,auuu = spl.cg(M,rhs,tol=1e-7)
             proj.vector()[:] = val      
         return  proj
     
    @staticmethod
    def Initialerror(u1, V,ArgsDic):
          e0 = Function(V)
          u = TrialFunction(V)
          v = TestFunction(V)
          V = ArgsDic['Hh'][2]
          n = FacetNormal(ArgsDic['mesh'])
          h = CellDiameter(ArgsDic['mesh'])
          dg = DGForms(n,h,ArgsDic)     
          m =  dg.gh_dar(u,v,grad(u),grad(v),Constant(1))
          Ge = assemble(m)
          Ge_mat = as_backend_type(Ge).mat()
          del Ge
          Ge = sp.csc_matrix(sp.csr_matrix(Ge_mat.getValuesCSR()[::-1], shape = Ge_mat.size))
          rhs = assemble(dg.bh_dar(u1,v))  # bh_prima or gh
          BigGe= spl.splu(Ge)
          e0.vector()[:] = BigGe.solve(np.array(rhs))    
          Gerror = sqrt(np.sum(np.array(np.abs(assemble(dg.gh(e0,e0,grad(e0),grad(e0),Constant(1)))))))
          del rhs,Ge
          return  Gerror,e0   
 
