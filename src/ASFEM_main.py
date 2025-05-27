# 2025 Juan Felipe Giraldo. Licensed under the MIT License.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

In this program we create a complete framework  for Lineal, NonLineal, Steady and Unsteady problem for VMS framework using 
and Adaptive Stabilized finite element method.
More information refer to:
    
Giraldo, J. F., & Calo, V. M. (2023).
A variational multiscale method derived from an adaptive stabilized conforming finite element method via residual minimization on dual norms. 
Computer Methods in Applied Mechanics and Engineering, 417, 116285.
https://doi.org/10.1016/j.cma.2023.116285
"""

import fenics
import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
import os, sys, time, subprocess, shutil
import sympy as spm
from sympy import init_printing
import pylab
import scipy as sc
import sympy as spm
from os.path import exists
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from mpltools import annotation
from dg_forms import DGForms, OptimalStabilization
from solvers_asfem import Solvers
from function_space_utils import FunctionSpaceUtils as fsu
from add_default_parameters import addDefaultParameters
from mesh_generator import MeshGenerator,BoundaryCreator, BoundaryMeasures    
from matrix_assembler import MatrixAssembler
from newton_solver import NewtonSolver
from refinement import RefinementStrategy
from time_integrator_method import TimeIntegratorMethod
from file_manager import FileManager
from postprocess import Postprocessor

set_log_active(False)
init_printing(forecolor='White')
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
 

def runSimulation(problem,params):
    
  text1=' NON-LINEAR '   if params['Nonlinear']  else ' LINEAR ';
  text2=' UNSTEADY'   if params['Unsteady']  else ' STEADY ';
         
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  print('SOLVING',text1,text2,params['Case']) 
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')     
    
  ''' Run steady state flow simulation with given parameters '''
  mesh = params['mesh']

  t0 = time.time()

  params = addDefaultParameters(params)
  params['foldername_solution'] = params['foldername_sol']
  folsolve = params['foldername_solution']

  iterR = 0  ; refi = 1 ;nord =0; iter_total=0; iter_totalDG =0; q=None

  u_k_finDG = None  ; u_primaDG= None;  u_newDGvec=None; u_newDGvec1=None  ;u_newDGvec2=None ; usolCG = None; Qsol = None;  dxGOA = None ; E_star= None; uexa = None ; grad_exa= None ; u_primaDG1= None;
  uDG_n = None ; uDG_n2 = None
  E = 0.1    ; g = 0
  
  boundaries = params['boundaries']

  gD_tags,gN_tags,gN2_tags,gR_tags = problem.BoundaryTags(params)

  bm = BoundaryMeasures(mesh, boundaries, params, gD_tags, gN_tags, gN2_tags, gR_tags)
  ds, dS, dsD, dsN, dsN2, dsR = bm.get_measures()


  dim = mesh.topology().dim()

  # Forcing term
  TimeSteps = 1 ; tini = params['tiniPM'];   #Stationary Case
  if params['Unsteady']:
         TS = params['TIM-PM']
         TimeSteps = int((params['tfinalPM']-params['tiniPM'])/params['dt-PM']+1)  #time steps
         Tvector = np.linspace(tini,params['tfinalPM'],TimeSteps)
         tini =  Tvector[params['ts_ini']]; 
         
  flag1 = 0;

  gD,gN     = problem.BoundaryCondition(tini,params)
  gR_alpha, gR_u = 0,0
  if params['RobinBC']:
     gR_alpha, gR_u = problem.RobinBoundaryCondition()

  f = problem.forcing_term(tini,mesh,params)

  V1,V2,V = fsu.get_FunctionSpaces(mesh,params)
  DOFset = fsu.get_spaceDofs(V)

  Hh = [V1,V2,V]
  
  dxini = Measure('dx') ; 

  opt = OptimalStabilization(mesh, dxini, ds, dS)
  eta = opt.compute_eta(params)

  #eta = (params['pdegree']+1)*(params['pdegree']+dim)/dim * opt.compute() * params['eta_0']

  boundT = 1
  
  beta_vec = problem.beta_vect_function(mesh,params)
  gammac   = problem.gamma_function(mesh,params)
  kappac   = problem.kappa_function(mesh,params)  

  gammaN, betaN, kappaN, beta_inf = problem.Norm_coefficients(mesh)     

  ArgsDic = {'Hh':Hh,'mesh':mesh,'DOFset': DOFset,'dxGOA':dxGOA,'f':f,'iterR':iterR, 
             'dx':dxini,'dS':dS,'ds':ds,'dsD':dsD,'dsN': dsN,'dsN2': dsN2,'dsR': dsR,
             'gD':gD,'gN':0,'gR_alpha':gR_alpha,'gR_u':gR_u,
             'gD_tags':gD_tags,'gN_tags':gN_tags, 'gN2_tags':gN2_tags, 'gR_tags':gR_tags,           
             'kappa':kappac,'kappa_norm':kappaN,
             'beta_inf':beta_inf,'beta_vec':beta_vec,'beta_norm':betaN,
             'gammac':gammac,'gamma_norm':gammaN,
             'eta':eta,'epsilon':params['epsilon'],'eta_adv':params['eta_adv'],         
             'Unsteady':params['Unsteady'],
             'NLterm':problem,'Nonlinear':params['Nonlinear'],'NeumannVar':params['NeumannVar'],
             'vect':params['vect'],'folsolve':folsolve,'boundT':boundT,'psuperc':params['superpen'],
             'automaticDIF':params['automaticDIF'], 'automaticDIF_DG':params['automaticDIF_DG'],'Solver':params['Solver'],
             "Ptype":params["Ptype"],"Gtype":params["Gtype"],'Stype':params["Stype"],
             'Tol_gram': params['Tol_gram'], 'Tol_schur': params['Tol_schur'], 
             'max_iter_cg' :params['max_iter_cg'], 'tol_ave' :params['tol_ave'],
             'ilu_drop_tol':params['ilu_drop_tol'], 'ilu_fill_factor':params['ilu_fill_factor']}
    
  if ArgsDic['Unsteady']:
         print('---------------------------UNSTEADY PROBLEM ------------------------------------')
         if params['ts_ini'] == 0:
             u_n = problem.Initial(params['tiniPM'],params) #u_ini
             u_np = problem.Initial(params['tiniPM'],params) #u_ini
             u_np2 = Constant(0) #u_ini
             uDG_np = problem.Initial(params['tiniPM'],params) #u_ini
             uDG_np2 = Constant(0) #u_ini

         elif params['importxml']:
              u_n,u_np,u_np2,uDG_np,uDG_np2 =  FileManager(params).load_solutions()
                         
         V1,V2,V = fsu.get_FunctionSpaces(mesh,params) #V1 discontionus, V2 contionus
         sol0 = Function(V)
         e_k0, u_k0 = sol0.split(True)
         u_k0   = project(u_n, V2) #only for NL porpouse (to improve)
         u_k0dg = project(u_n, V1)
         post = Postprocessor(ArgsDic, params)
  
         namesol  = 'sol-t'+str(tini)+'-r'
         if params['Solpvd'] : post.plota(u_k0, namesol ,0, 'pvd', folsolve)

         C1be,C2be,Crhs1be,Crhs2be,Crhs3be = TimeIntegratorMethod('BE',params).coefficients() #it starts with BE to compute the first time step
         UnsteadyDic = {'dt':params['dt-PM'],\
                        'C1': C1be,'C2': C2be,'Crhs1': Crhs1be,\
                        'Crhs2': Crhs2be,'Crhs3': Crhs3be,\
                        'u_n': u_n,'u_n2': Constant(0), 'TIM':'BE', 't':tini} #initialized for T0 (solve L2 for u0)
         ArgsDic.update(UnsteadyDic)  #check u_n=u_n otherwise u_n:u_k0 check V2  problems for non_linear cases
         folsolveini = params['foldername_solution']+'/time_ini'
         
         err0, errplot=fsu.Initialerror(u_k0-u_k0dg, V1, ArgsDic)
         print('err0 =',err0)
         post = Postprocessor(ArgsDic, params)

         if (params['Solpng'] and params['Exact']): post.plota(errplot,'Initialerror',0,'png',folsolveini)

         if (params['Solpng'] and params['Exact']): post.plota(u_k0,'sol-tini',0,'png',folsolveini)
         if params['Solpng'] == True: post.plota(u_k0,'sol-exaini',0,'png',folsolveini)
         if params['Solpvd'] == True: post.plota(u_k0,'sol-exaini',0,'pvd',folsolveini)

         if TS == 'BDF2' : #initialize the t(n+1) with  BE
             nord = 2 
             t2 = Tvector[params['ts_ini']+1]

             gD2,gN2 = problem.BoundaryCondition(t2,params)
             f2 = problem.forcing_term(t2,mesh,params)
             ArgsDic.update({'gD':gD2,'gN':gN2,'f':f2})
             if params['ts_ini']==0:
                 if ArgsDic['Nonlinear'] :
                     sol = Function(V)
                     e_k, u_k = sol.split(True)
                     e_k.vector()[:] *= 0
                     ns = NewtonSolver(params, ArgsDic)
                     [e2, u_k, iterN0] = ns.Damping_solve(e_k, u_k0, V) 
                     if params['DG']:[u_kDG,iterNDG] = ns.Damping_solve_DG(u0, V) 
     
                 else:
                      e,u,v,w = fsu.get_TestTrial(V,ArgsDic)
                      a,l = MatrixAssembler(ArgsDic).system_matrices_asfem(Constant(0),Constant(0), e, u, w, v)

                      e2 , u_k = Solvers(ArgsDic).solverfun(a,l,V,params['Solver'])
             else: 
                    e2 = Constant(0); iterN0=0
                    u_k = interpolate(u_np2,V1)


             folsolve2 = params['foldername_solution']+'/timeBE'+str(t2)
             uexa,grad_exa = problem.exact_solution(t2,params)
             if params['grad_exa']:  Usol_exa = fsu.projection_exa(uexa,grad_exa,V2,'H1',0,ArgsDic)
             else:  Usol_exa = interpolate(uexa,V2)     
             post = Postprocessor(ArgsDic, params)

             if (params['Solpng'] and params['Exact']) : post.plota(Usol_exa,'sol-exat1-BE',0,'png',folsolve2)
             if params['Solpng']: post.plota(u_k,'sol-t1-BE',ArgsDic['iterR'],'png',folsolve2)
             if params['Solpvd']: post.plota(u_k,'sol-t1-BE',ArgsDic['iterR'],'pvd',folsolve2)

             C1,C2,Crhs1,Crhs2,Crhs3 = TimeIntegratorMethod('BDF2',params).coefficients() #it starts with BE to compute the first time step
             UnsteadyDic.update({'C1': C1,'C2': C2,'Crhs1': Crhs1, 'Crhs2': Crhs2,'Crhs3': Crhs3,\
                                'u_n': u_n,'u_n2': u_k,'TIM':'BDF2'}) #initialized for T0 (solve L2 for u0)
                 
                 
             ArgsDic.update({'gD':gD,'gN':gN,'f':f})
             ArgsDic.update(UnsteadyDic)
             namesol  = 'sol-t'+str(t2)+'-r'
             Postprocessor(ArgsDic, params).plota(u_k, namesol ,0, 'pvd', folsolve)


         elif TS == 'BE':
              nord = 1
              sol = Function(V)
              e_k, u_k = sol.split(True)
              e_k.vector()[:] *= 0
              u_k.vector()[:] = u_k0.vector()[:]
              if params['DG']: 
                  u_kDG = Function(V1)
                  u_kDG.vector()[:] = u_k0dg.vector()[:]

  else:
     t = 0 
     if params['Nonlinear']:
         print('------------------------NON-LINEAR DAR STATIONARY PROBLEM ------------------------------------')
         u_n = problem.Initial(0,params) #u_ini
         V1,V2,V = fsu.get_FunctionSpaces(mesh,params) #V1 discontionus, V2 contionus
         sol0 = Function(V)
         e_k, u_k = sol0.split(True)
         e_k.vector()[:] *= 0
         u_k = interpolate(u_n,  V2)
         if params['DG']: u_kDG = interpolate(u_n,V1)#interpolate(u_n, V1)

         if params['Exact'] == True: Postprocessor(ArgsDic, params).plota(u_k,'sol2D-iniguess',0,'png',folsolve)      #plot initial guess in mesh 0
     else:
         print('------------------------LINEAR DAR STATIONARY PROBLEM ------------------------------------')
         e_k = Constant(0)
         u_k = Constant(0)

  for time_step in range(params['ts_ini']+nord,TimeSteps):
      flagNL = 0
      ArgsDic['Time_step']=time_step 
      NormDic={'L2errCG':[],'L2errDG':[],'L2errnewDG':[],'L2errnewDG1':[],'L2errnewDG2':[],
               'VherrCG':[],'VherrDG':[],'VherrnewDG':[],'VherrnewDG1':[],'VherrnewDG2':[],
               "Dofsvect":[],"E":[],"E_GOA":[],'Vherr_uprima_DG':[],'hmin':[]}

      if ArgsDic['Unsteady']:
         print("TIME STEP PM:"+str(time_step))
         refi = 1; iterR=0;
         if time_step > params['ts_ini']+nord:
             if params['TIM-PM'] == 'BDF2':  
                 u_n= u_fin0 ;  u_n2= u_fin
                 if params['DG']:
                       uDG_n= uDG_fin0 ; uDG_n2 = uDG_fin                       
             elif params['TIM-PM'] == 'BE': 
                 u_n= u_fin ;   u_n2= Constant(0)
                 if params['DG']:
                       uDG_n= Function(V1)
                       uDG_n.assign(uDG_fin) ; uDG_n2 = Constant(0)
             
             bm = BoundaryMeasures(mesh, boundaries, params, ArgsDic['gD_tags'],ArgsDic['gN_tags'],ArgsDic['gN2_tags'],ArgsDic['gR_tags'])
             ds, dS, dsD, dsN, dsN2, dsR = bm.get_measures()

             ArgsDic.update({'u_n-2':u_n2, 'u_n-1':u_n, 'uDG_n-2':uDG_n2, 'uDG_n-1':uDG_n})
                 
         t = Tvector[time_step]
         UnsteadyDic.update({'t': t})
         params.update({'t': t})

         gD,gN = problem.BoundaryCondition(t,params)
         f = problem.forcing_term(t,mesh,params)

         folsolve = params['foldername_solution']+'/time'+str(t)

         ArgsDic.update({'gD':gD,'gN':gN,'f':f,'t': t,'folsolve':folsolve,'NeumannVar':params['NeumannVar']})

         print('TIME STEP=',time_step,'time=',t, 'dt=',params['dt-PM'],'Final_time=', params['tfinalPM'])

      while (iterR < params['MAX_REF']) and (refi==1):
          
        f = problem.forcing_term(t,mesh,params)
        if params['q_deg']: 
           dx = Measure('dx', metadata={"quadrature_scheme":"default",
                                        "quadrature_degree":params['q_deg']}, domain = mesh)
           bm = BoundaryMeasures(mesh, boundaries, params, ArgsDic['gD_tags'],ArgsDic['gN_tags'],ArgsDic['gN2_tags'],ArgsDic['gR_tags'])
           ds, dS, dsD, dsN, dsN2, dsR = bm.get_measures()
           ArgsDic.update({'ds':ds,'dS':dS,'dsD':dsD,'dsN':dsN,'dsN2':dsN2,'dsR':dsR })      
        else: 
           dx = Measure('dx', domain=mesh) 
                
        opt = OptimalStabilization(mesh, dx, ds, dS)
        eta = opt.compute_eta(params)

        V1,V2,V = fsu.get_FunctionSpaces(mesh,params)
        
        if params['Exact']:
            uexa,grad_exa = problem.exact_solution(t,params)
            if params['post_end'] == 0: Postprocessor(ArgsDic, params).plotexa(uexa,V2,iterR,folsolve)

        DOFset = fsu.get_spaceDofs(V)

        hmin = mesh.hmin()
        hmax = mesh.hmax()

        Hh = [V1,V2,V]

        print('--------------------------------')
        print('--------------------------------')
        print('level =',iterR,', Ndofse =', DOFset[0], ', Ndofsu =', DOFset[1], ', Ndofs total =', DOFset[2])
        print('h min = %s, h max = %s' %(hmin, hmax))
        print('--------------------------------')
        print('--------------------------------')

        # Get Solution of the primal problem (ASFEM)
        ArgsDic.update({'kappa':kappac, 'kappa_norm':kappaN,
                        'beta_inf':beta_inf,'beta_vec':beta_vec,'beta_norm':betaN,
                        'f':f,'Hh':Hh,'mesh':mesh,'ds':ds,'dS':dS,'dx':dx,
                        'dsD':dsD,'dsN': dsN,'dsN2': dsN2,'dsR': dsR,'dxGOA':dxGOA,
                        'DOFset': DOFset,'eta':eta,'iterR':iterR})

        if  ArgsDic['Unsteady']: 
             
          if  params['REF_TYPE']!=0 and time_step > params['ts_ini']+nord:
                   u_np  = fsu.projection(ArgsDic['u_n-1'], V2, 'Vh', 0, ArgsDic) ;         
                   #u_np  = fsu.projection(ArgsDic['u_n-1'], V2, 'L2', 0, ArgsDic) ;         
                   if params['TIM-PM'] == 'BDF2':   
                       u_np2 = fsu.projection(ArgsDic['u_n-2'], V2, 'Vh', 0, ArgsDic)
                       #u_np2 = projection(ArgsDic['u_n-2'], V2, 'L2', 0, ArgsDic)
         
                   if params['DG']: 
                      uDG_np  = fsu.projection(ArgsDic['uDG_n-1'], V1, 'Vh', 0 ,ArgsDic) ;     
                      #uDG_np  = fsu.projection(ArgsDic['uDG_n-1'], V1, 'L2', 0 ,ArgsDic) ;         

                      if params['TIM-PM'] == 'BDF2':  
                          uDG_np2 = fsu.projection(ArgsDic['uDG_n-2'], V1, 'Vh', 0 ,ArgsDic)
                          #uDG_np2 = fsu.projection(ArgsDic['uDG_n-2'], V1, 'L2', 0 ,ArgsDic)
              
                   if ArgsDic['Nonlinear'] :
                       
                        sol = Function(V)
                        e_k, u_k = sol.split(True)
                        e_k.vector()[:] *= 0

                        if flagNL ==0:
                            e_k.vector()[:] *= 0
                            if TS == 'BDF2':  
                                u_k.vector()[:] = u_np2.vector()[:]
                                if params['DG']: 
                                  u_kDG = Function(V1) 
                                  u_kDG.vector()[:] = uDG_np2.vector()[:]
                                #u_np2 used as an initial guess (u_k)to solve u in the second Time step.
                            elif TS == 'BE':
                                u_k.vector()[:] = u_np.vector()[:]
                                if params['DG']:
                                  u_kDG = Function(V1)  
                                  u_kDG.vector()[:] = uDG_np.vector()[:]  
                        else: 
                             e_k = interpolate(e_k_fin, V1)
                             u_k = interpolate(u_k_fin, V2)
                             if params['DG']:
                                 u_kDG = interpolate(u_k_finDG, V1)
                             flagNL =0          
  
          if time_step == params['ts_ini']+nord and flag1 ==1: #or time_step > params['ts_ini']+nord :
                  if TS == 'BDF2':                     
                      ArgsDic['TIM']='BE'
                      ArgsDic.update({'C1': C1be,'C2': C2be,'Crhs1': Crhs1be,'Crhs2': Crhs2be,'Crhs3': Crhs3be,'u_n2': Constant(0)})
                      #ei, ui = TrialFunctions(V); wi, vi = TestFunctions(V)
                      if ArgsDic['Nonlinear']:
                          sol = Function(V)
                          e_k, u_k = sol.split(True)
                          e_k.vector()[:] *= 0
                          ns = NewtonSolver(params, ArgsDic)
                          [e2, u_np2, iterN0] = ns.Damping_solve(e_k, u_k, V) 
                          if params['DG']: 
                             uDG_np2 = fsu.projection(u_np2, V1, 'Vh', 0 ,ArgsDic)
                      
                      else: 
                          ei,ui,vi,wi = fsu.get_TestTrial(V,ArgsDic)
                          ai,li = MatrixAssembler(ArgsDic).system_matrices_asfem(Constant(0),Constant(0), ei, ui, wi, vi)

                          e2i , u_np2 = Solvers(ArgsDic).solverfun(ai,li,V,params['Solver'])    
                      
                          uDG_np2 = Constant(0)
                          if params['DG']: 
                             uDG_np2 = fsu.projection(u_np2, V1, 'Vh', 0 ,ArgsDic)
                      
                      ArgsDic.update({'C1': C1,'C2': C2,'Crhs1': Crhs1,'Crhs2': Crhs2,'Crhs3': Crhs3,'u_n2':u_np2, 'uDG_n2':uDG_np2})
                      ArgsDic['TIM']='BDF2'
                      post=Postprocessor(ArgsDic, params)
                      if params['Solpng']: post.plota(u_np2,'sol-t1-BE',ArgsDic['iterR'],'png',folsolve2)
                      if params['Solpvd']: post.plota(u_np2,'sol-t1-BE',ArgsDic['iterR'],'pvd',folsolve2)

                  if ArgsDic['Nonlinear'] :
                       sol = Function(V)
                       e_k, u_k = sol.split(True)
                       u_np  = fsu.projection(ArgsDic['Usol'], V2, 'Vh', 0, ArgsDic) ; 
                       if params['DG']: 
                           uDG_np   = project(ArgsDic['UsolDG'], V1) ; 
                        
                       if flagNL ==0:
                           e_k.vector()[:] *= 0
                           if TS == 'BDF2':  
                               u_k.vector()[:] = u_np2.vector()[:]
                               if params['DG']: 
                                 u_kDG = Function(V1) 
                                 u_kDG.vector()[:] = uDG_np2.vector()[:]
                               #u_np2 used as an initial guess (u_k)to solve u in the second Time step.
                           elif TS == 'BE':
                               u_k.vector()[:] = u_np.vector()[:]
                               if params['DG']:
                                 u_kDG = Function(V1)  
                                 u_kDG.vector()[:] = uDG_np.vector()[:]
                                                      
                       else: 
                           e_k = interpolate(e_k_fin, V1)
                           u_k = interpolate(u_k_fin, V2)
                           if params['DG']:
                               u_kDG = interpolate(u_k_finDG, V1)
                           flagNL =0 

          flag1 = 1

        else: #stationary
                  if ArgsDic['Nonlinear'] :
                      if flagNL == 0:
                         e_k.vector()[:]*= 0.0
                         flagNL = 1
                      else:
                         sol0 = Function(V)
                         e_k, u_k = sol0.split(True)
                         e_k = interpolate(e_k_fin, V1)
                         u_k = interpolate(u_k_fin, V2)
                         if params['DG']:

                             u_kDG = fsu.projection(u_k_finDG, V1, 'Vh', 0 ,ArgsDic)

        if  ArgsDic['Nonlinear']:
            
            if  ArgsDic['Unsteady'] and time_step >  params['ts_ini']+nord:
                ArgsDic.update({'u_n':u_np, 'u_n2':u_np2 }) #CHeck if nonlinear V2 if problems
            
            ns = NewtonSolver(params, ArgsDic)
            [e_k_fin,u_k_fin,iterN] = ns.Damping_solve(e_k, u_k, V) 

            uDGprueb = fsu.disc_proj_Uh(u_k, V1, mesh.topology().dim())

            usolCG = fsu.disc_proj_Uh(u_k_fin, V1, mesh.topology().dim())
            
            uCG = Function(V2)
            uCG.vector()[:] = u_k_fin.vector()[:]
            u_primaDG1 = None

            if params['DG']:
                 [u_k_finDG,iterNDG] = ns.Damping_solve_DG(u_kDG,V1) 
                 iter_totalDG += iterNDG
           
            if params['MS']:
                ud = TrialFunction(V1); wd = TestFunction(V1)
                ams,lms1,lms2 = MatrixAssembler(ArgsDic).system_matrices_ms(usolCG,e_k_fin,ud,wd)
                
                ek0, u_primaDG = Solvers(ArgsDic).solverfun(ams,lms1+lms2,V1,params['SolverDG'])
                u_newDGvec = Function(V1)
                u_newDGvec.vector()[:] =usolCG.vector()[:] + u_primaDG.vector()[:]
                
                if params['MSupdate']:
                  u_k_fin = Function(V1)
                  u_k_fin.vector()[:] = u_newDGvec.vector()[:]
                
            if params['MS1']:
                ek0, u_primaDG1 = Solvers(ArgsDic).solverfun(ams,lms1,V1,params['SolverDG'])
                u_newDGvec1 = Function(V1)
                u_newDGvec1.vector()[:] =usolCG.vector()[:] + u_primaDG1.vector()[:]
                
            if params['MS2']:
                
                ek0, u_primaDG2 = Solvers(ArgsDic).solverfun(ams,lms2,V1,params['SolverDG'])
                u_newDGvec2 = Function(V1)
                u_newDGvec2.vector()[:] =usolCG.vector()[:] + u_primaDG2.vector()[:]

            iter_total += iterN

        else:
            
            if  ArgsDic['Unsteady']: ArgsDic.update({'u_n':u_np, 'u_n2':u_np2 }) #CHeck if nonlinear V2 if problems
            
            e,u,v,w = fsu.get_TestTrial(V,ArgsDic)
            ma = MatrixAssembler(ArgsDic)
            a,l = ma.system_matrices_asfem(Constant(0),Constant(0), e, u, w, v)

            n = FacetNormal(ArgsDic['mesh'])
            h = CellDiameter(ArgsDic['mesh'])         
     
            e_k_fin,u_k_fin = Solvers(ArgsDic).solverfun(a,l,V,params['Solver'])
            
            uCG = Function(V2)
            uCG.vector()[:] = u_k_fin.vector()[:]
            
            usolCG = fsu.disc_proj_Uh(u_k_fin, V1, mesh.topology().dim())

            u_k_fin = Function(V1)
            u_k_fin.vector()[:] = usolCG.vector()[:]
            
            if params['DGprojection'] :
                 ud = TrialFunction(V1); wd = TestFunction(V1)
                               
                 n = FacetNormal(ArgsDic['mesh'])
                 h = CellDiameter(ArgsDic['mesh'])
                 dg = DGForms(n,h,ArgsDic)

                 lms  =  dg.gh(e_k_fin,wd,grad(e_k_fin),grad(wd),Constant(1)) + dg.bh_prima(u_k,e_k_fin,wd) 
                 ams   =  dg.bh_prima(u_k,ud,wd)
     
                 ek0,u_exDG  = Solvers(ArgsDic).solverfun(ams,lms,V1,params['SolverDG'])

                 u_solDisc = Function(V1)
                 u_solDisc.vector()[:] = usolCG.vector()[:] + u_exDG.vector()[:]
                 u_k_fin.vector()[:]   = u_solDisc.vector()[:]
                  
            q = []
                
            if params['DG']:
                if  ArgsDic['Unsteady']: ArgsDic.update({'u_n':uDG_np, 'u_n2':uDG_np2 }) #CHeck if nonlinear V2 if problems
                ud = TrialFunction(V1); wd = TestFunction(V1)
                ma = MatrixAssembler(ArgsDic)
                adg,ldg = ma.system_matrices_dg(Constant(0), ud, wd)

                ek0,u_k_finDG = Solvers(ArgsDic).solverfun(adg,ldg,V1,params['SolverDG'])

            if params['MS']:
                ud = TrialFunction(V1); wd = TestFunction(V1)
                ams,lms1,lms2 = MatrixAssembler(ArgsDic).system_matrices_ms(0,e_k_fin,ud,wd)

                ek0,u_primaDG  = Solvers(ArgsDic).solverfun(ams,lms1+lms2,V1,params['SolverDG'])

                u_newDGvec = Function(V1)
                u_newDGvec.vector()[:] =usolCG.vector()[:] + u_primaDG.vector()[:]
                
                if params['MSupdate']:
                  u_k_fin = Function(V1)
                  u_k_fin.vector()[:] = u_newDGvec.vector()[:]
                
            if params['MS1']:
                ek0,u_primaDG1 = Solvers(ArgsDic).solverfun(ams,lms1,V1,params['SolverDG'])

                u_newDGvec1 = Function(V1)
                u_newDGvec1.vector()[:] =usolCG.vector()[:] + u_primaDG1.vector()[:]
                
            if params['MS2']:
                
                ek0, u_primaDG2 = Solvers(ArgsDic).solverfun(ams,lms2,V1,params['SolverDG'],ArgsDic)

                u_newDGvec2 = Function(V1)
                u_newDGvec2.vector()[:] =usolCG.vector()[:] + u_primaDG2.vector()[:]    
                                

            iter_total += 1
            
        rs = RefinementStrategy(ArgsDic, params)
        [E,g,g_plot]  = rs.compute_error(e_k_fin)

        SolDic = {"Usol":u_k_fin,"UsolCG": uCG, "Esol":e_k_fin,"Udis":u_k_fin,"UsolDG":u_k_finDG,"gplot":g_plot,
                  "uexa":uexa,"gradexa":grad_exa,"E":E,"E_star":E_star,'u_primaDG':u_primaDG1, 'SolDGnew': u_newDGvec,
                  'UsolCG-dis': usolCG, 'UsolDGnew': u_newDGvec,'UsolDGnew1': u_newDGvec1, 'UsolDGnew2': u_newDGvec2 ,"Qsol":q }
        
        ArgsDic.update(SolDic)
        post = Postprocessor(ArgsDic, params)

        if ArgsDic['Nonlinear']:
          [r_norm_fin, r_norm_fin_2] = post.postprocess_NLresiduos(SolDic)

        else:
          r_norm_fin_2 = SolDic["E"]

        if params['post_end'] == 0: post.postprocess_plots(SolDic)

        if ArgsDic['Unsteady'] == False or (ArgsDic['Unsteady'] and params['REF_TYPE']==0)  or (ArgsDic['Unsteady'] and ArgsDic['iterR'] > 0 and params['REF_TYPE']!=0 ): 
            NormDic = post.postprocess_norms(SolDic,NormDic)

            if ArgsDic['iterR'] > 0 : post.postprocess_convergence_space(NormDic,ArgsDic['Time_step'])
            if ArgsDic['Unsteady'] == False: np.save(params['foldername_con']+'/NormDic-space-p'+str(params["pdegree"])+'.npy', NormDic)
            else:
                np.save(params['foldername_con']+'/NormDic-space-p'+str(params["pdegree"])+'-ts'+str(ArgsDic['Time_step']) +'.npy', NormDic)


        if (iterR == params['MAX_REF']-1) or ((r_norm_fin_2 < params['TOL']) and (params['TrialType'] != 'DG')) or (DOFset[2] > params['dofmax']):
          if (E < params['Cref'] and (params['TrialType']  != 'DG')): print("Stop by tolerance in the residual norm")
          elif  DOFset[2] > params['dofmax']:              print("Stop by error Max number of dofs")
          elif  iterR == params['MAX_REF']-1:              print("Stop by error Max number of refinements")

          NormDic_exp = NormDic
          if  ArgsDic['Unsteady']:

             #if params['MS']:  u_fin = Function(V1) 
             if params['MSupdate']: u_fin = Function(V1)

             if 1 : u_fin = Function(V1)
             #else:  u_fin = Function(V2) 
             
             u_fin.vector()[:]  = u_k_fin.vector()[:]
             
             if params['DG']:
                 uDG_fin = Function(V1) 
                 uDG_fin.vector()[:]  = u_k_finDG.vector()[:]
                 del u_k_finDG

             if TS=='BDF2':#(change to fun V1 - DG projection from the previous TS)
               u_fin0 = Function(V2)              
               u_fin0.vector()[:] = u_np2.vector()[:]
               if params['DG']:
                 uDG_fin0 = Function(V1)              
                 uDG_fin0.vector()[:] = uDG_np2.vector()[:]

             if params["xmlfiles"] : 
                 post = Postprocessor(ArgsDic, params)
                 post.save_object(mesh, 'mesh',iterR ,params['foldername_solution']+'/timexml','xml')
                 post.save_object(u_fin, 'sol',iterR ,params['foldername_solution']+'/timexml','xml')
             if params['DG'] and params["xmlfiles"] : post.save_object(uDG_fin, 'solDG', ArgsDic, params,iterR ,params['foldername_solution']+'/timexml','xml')
 
             mesh = MeshGenerator(params).Create_mesh()
             boundaries  = BoundaryCreator(params,mesh).create_boundaries()
             
             if params["timepvd"]  : post.save_object(u_fin,'sol',iterR ,params['foldername_solution']+'/timepvd','pvd')

             bm = BoundaryMeasures(mesh, boundaries, params, ArgsDic['gD_tags'],ArgsDic['gN_tags'],ArgsDic['gN2_tags'],ArgsDic['gR_tags'])
             ds, dS, dsD, dsN, dsN2, dsR = bm.get_measures()
    
          print('Finished in {0:.2f} seconds'.format(time.time()-t0))
          refi=0
          if params['post_end'] == 1:
              post=Postprocessor(ArgsDic, params)
              post.postprocess_plots(SolDic)
              if params['Exact']: post.plotexa(uexa,V2,iterR,folsolve)

        else:
          rs = RefinementStrategy(ArgsDic, params)
          mesh = rs.refine_mesh(mesh, E, g)

          boundaries  = BoundaryCreator(params,mesh).create_boundaries()
          
          bm = BoundaryMeasures(mesh, boundaries, params, ArgsDic['gD_tags'],ArgsDic['gN_tags'],ArgsDic['gN2_tags'],ArgsDic['gR_tags'])
          ds, dS, dsD, dsN, dsN2, dsR = bm.get_measures()
          iterR +=1

      print(params['Case'])
      print('TOTAL ITERATIONS ALL LEVELS = ', iter_total)

      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
      print('h min = %s, h max = %s' %(hmin, hmax))
      print('kappa %s,  Pol= %s, dt= %s' %(ArgsDic['kappa'], params["pdegree"],params['dtlist'][0]))
      if params['Unsteady']:
          print('TIM: ',params['TIM-PM'] )

      print('NUMBER OF LEVELS =', iterR)
      print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


  return SolDic
