#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 20:50:32 2025

@author: Juan Giraldo
"""
import time
import numpy as np
from dolfin import (Function, TrialFunction, TestFunction, interpolate, Constant,
                   FacetNormal, CellDiameter, sqrt, assemble, grad)
from solvers_asfem import Solvers
from dg_forms import DGForms
from matrix_assembler import MatrixAssembler
from function_space_utils import FunctionSpaceUtils as fsu
# ----------------------------------------------------------------------#
# Newton Solution
# ----------------------------------------------------------------------#
class NewtonSolver:
    def __init__(self, params, argsDic):
        self.params = params
        self.argsDic = argsDic

    def Damping_solve(self, e_k, u_k, V):     
         mesh = V.mesh()
         V1= V.sub(0).collapse()
         e,u,v,w = fsu.get_TestTrial(V,self.argsDic)

         iter = 1
         K = 0
         n_failed = 0
         convergencia = 0
         solver = 1
         alpha         = self.params['alpha_d']
         error_delta_e = self.params['error_delta_e']
         error_delta_u = self.params['error_delta_u']
         maxiter       = self.params['maxiter_d']
         omega         = self.params['omega_d']
         factor        = self.params['factor_d']
         tol_newton    = self.params['tol_newton']
         #
         e_kmen1 = interpolate(Constant(0.0), V1)
         
         gammaN =    self.argsDic['NLterm'].Norm_coefficients(mesh)[0] 
         betaN =     self.argsDic['NLterm'].Norm_coefficients(mesh)[1]
         kappaN =    self.argsDic['NLterm'].Norm_coefficients(mesh)[2] 
         betainf =   self.argsDic['NLterm'].Norm_coefficients(mesh)[3] 

         if gammaN:  self.argsDic.update({'gamma_norm' : gammaN})
         if betaN:   self.argsDic.update({'beta_norm'  : betaN})
         if kappaN:  self.argsDic.update({'kappa_norm' : kappaN})

         self.argsDic.update({'beta_inf' : betainf})
         
         n = FacetNormal(mesh)
         h = CellDiameter(mesh)
         dg = DGForms(n,h,self.argsDic)

         norm_res_ini = fsu.normresidual(e_k,u_k,w,v,self.argsDic)
         ma = MatrixAssembler(self.argsDic)

         while (convergencia ==0) and (iter < maxiter) and (norm_res_ini > tol_newton):
             if solver == 1:
                 norm_res_k = fsu.normresidual(e_k,u_k,w,v,self.argsDic)
                 a,l = ma.system_matrices_asfem(e_k, u_k, e, u, w, v)
                 #a,l = system_Matrices_ASFEM(e_k,u_k,e,u,w,v,self.argsDic)
                 de_k1,du_k1 = Solvers(self.argsDic).solverfun(a,l,V,self.params['Solver'])
                 sol1 = Function(V)
                 e_k1, u_k1 = sol1.split(True)
             solver = 0
             t_k = 1/(1 + K*norm_res_k)
             u_k1.vector()[:] = u_k.vector() + t_k*du_k1.vector()
             e_k1.vector()[:] = e_k.vector() + t_k*de_k1.vector()
             #
             norm_res_k1 = fsu.normresidual(e_k1,u_k1,w,v,self.argsDic)

             om_k = 1/t_k*(1 - norm_res_k1/norm_res_k)

             if om_k < omega and abs(om_k) > 1e-12:
                 n_failed += 1
                 if K == 0:
                     # print('~~~ K pasa de 0 a 1 ~~~')
                     K = 1
                 else:
                     # print('~~~ Agrandando K ~~~')
                     K = K*10
             else:
                 error_delta_u = sqrt(assemble(dg.gh(u_k1 - u_k,u_k1 - u_k,grad(u_k1 - u_k),grad(u_k1 - u_k),Constant(1))))
                 error_delta_e = sqrt(assemble(dg.gh(e_k1 - e_k,e_k1 - e_k,grad(e_k1 - e_k),grad(e_k1 - e_k),Constant(1))))

                 if (error_delta_u < tol_newton):
                     print('------- Has convergido! -------')
                     print('error delta_u =', error_delta_u)
                     print('error delta_e =', error_delta_e)
                     u_k = 1*u_k1
                     e_k = 1*e_k1
                     convergencia = 1
                     print('norm_res_k1 =', norm_res_k1)
                 else:
                     iter += 1
                     K = K/10
                     e_kmen1 = 1*e_k
                     u_k = 1*u_k1
                     e_k = 1*e_k1
                     solver = 1
         print('----------------------------------')
         print('----------------------------------')
         print('iterationsCG =', iter)
         sol_fin = Function(V)
         e_k_fin, u_k_fin = sol_fin.split(True)
         e_k_fin, u_k_fin = e_k, u_k
         return e_k_fin, u_k_fin  ,iter   
  
    def Damping_solve_DG(self, u_k, V):  
          #parameters
          mesh = self.argsDic['mesh']
          u = TrialFunction(V) ; w = TestFunction(V)
          iter = 1
          K = 0
          n_failed = 0
          convergencia = 0
          solver = 1
          
          alpha         = self.params['alpha_d']
          error_delta_e = self.params['error_delta_e']
          error_delta_u = self.params['error_delta_u']
          maxiter       = self.params['maxiter_d']
          omega         = self.params['omega_d']
          factor        = self.params['factor_d']
          tol_newton    = self.params['tol_newton']
    
          gammaN =    self.argsDic['NLterm'].Norm_coefficients(mesh)[0]  
          betaN =     self.argsDic['NLterm'].Norm_coefficients(mesh)[1] 
          kappaN =    self.argsDic['NLterm'].Norm_coefficients(mesh)[2] 
          betainf =   self.argsDic['NLterm'].Norm_coefficients(mesh)[3] 
    
          if gammaN:  self.argsDic.update({'gamma_norm' : gammaN})
          if betaN:   self.argsDic.update({'beta_norm'  : betaN})
          if kappaN:  self.argsDic.update({'kappa_norm' : kappaN})
    
          self.argsDic.update({'beta_inf' : betainf})
    
          n = FacetNormal(mesh)
          h = CellDiameter(mesh)
          dg = DGForms(n,h,self.argsDic)
    
          norm_res_ini = fsu.normresidualDG(u_k,w,self.argsDic)
          ma = MatrixAssembler(self.argsDic)

          while (convergencia ==0) and (iter < maxiter) and (norm_res_ini > tol_newton  ):
    
              if solver == 1:
                  norm_res_k = fsu.normresidualDG(u_k,w,self.argsDic)
                  #a,l = system_Matrices_DG(u_k,u,w,self.argsDic)
                  a,l = ma.system_matrices_dg(u_k, u, w)

    
                  eee,du_k1 = Solvers(self.argsDic).solverfun(a,l,V,self.params['SolverDG'])
              u_k1 = Function(V)
              solver = 0
              t_k = 1/(1 + K*norm_res_k)
              u_k1.vector()[:] = u_k.vector() + t_k*du_k1.vector()
    
              norm_res_k1 = fsu.normresidualDG(u_k1,w,self.argsDic)
    
              om_k = 1/t_k*(1 - norm_res_k1/norm_res_k)
    
              if om_k < self.params['omega_d']  and abs(om_k) > 1e-10:
                  n_failed += 1
                  if K == 0:
                      K = 1
                  else:
                      K = K*5
              else:
                  error_delta_u =  sqrt(assemble(dg.gh(u_k1 - u_k,u_k1 - u_k,grad(u_k1 - u_k),grad(u_k1 - u_k),Constant(1))))
    
                  if (error_delta_u < tol_newton  ):
                      print('------- Has convergido! -------')
                      print('error delta_uDG =', error_delta_u)
                      u_k = 1*u_k1
                      convergencia = 1
                      print('norm_res_k1DG =', norm_res_k1)
                  else:
                      iter += 1
                      K = K/10
                      u_k = 1*u_k1
                      solver = 1
          print('----------------------------------')
          print('----------------------------------')
          print('iterationsDG =', iter)
          u_k_fin = u_k
          return u_k_fin  ,iter