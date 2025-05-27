# 2025 Juan Felipe Giraldo. Licensed under the MIT License.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:21:41 2025

"""

def addDefaultParameters(params):

  ''' Add default parameters needed to run simulation
      @param params: dictionary of parameters from runFlowSimulation()
  '''
  if 'Case' not in params:
    params["Case"] = None
  if 'dim' not in params:
      params["dim"] = 2
  if 'Nonlinear' not in params:
      params["Nonlinear"] = False
  # ----------------------------------------------------------------------#
  #Temporal params
  # ----------------------------------------------------------------------#  
  if 'Unsteady' not in params:
      params["Unsteady"] = False   
  if 'tfinalPM' not in params:
    params["tfinalPM"] = 1.0
  if 'dtlist' not in params:
    params['dtlist'] = [1]
  if 'ts-imp' not in params:
    params['ts-imp'] = 10 
  if 't' not in params:
    params["t"] = 0
  if 'ts_ini' not in params:
    params["ts_ini"] = 0    
    
  # ----------------------------------------------------------------------#
  #Element space
  # ----------------------------------------------------------------------#
  if 'pdegree' not in params:
    params["pdegree"] = 1
  if 'TestType' not in params:
    params["TestType"] = "DG"
  if 'TrialType' not in params:
    params["TrialType"] = "CG"
  if 'Ptrial' not in params:
    params["Ptrial"] = params["pdegree"]
  if 'Ptest' not in params:
    params["Ptest"] = params["pdegree"]
  if 'superpen' not in params:
    params['superpen'] = 0  
  if 'automaticDIF' not in params:
      params['automaticDIF'] = 0
  if 'automaticDIF_DG' not in params:
      params['automaticDIF_DG'] = 0  
  if 'NeumannVar' not in params:
      params['NeumannVar'] = False 
      
  # ----------------------------------------------------------------------#
  # Iteration parameters
  # ----------------------------------------------------------------------#
  # Max number of iteration for adaptivity
  if 'MAX_REF' not in params:
    params['MAX_REF'] = 10
  # Stopping Criterion due to the Error norm estimation
  if 'Cref' not in params:
    params['Cref'] = 1e-18
  # Max number of dof before to stop the adaptive refinement
  if 'dofmax' not in params:
    params['dofmax'] = 1e6
  if 'TOL' not in params:
    params['TOL'] = 1e-8
  if 'eta_0' not in params:
      params['eta_0'] = 1.0
      
  # ----------------------------------------------------------------------#
  # Refinement parameters
  # ----------------------------------------------------------------------#
  # Refinement type: (1) Adaptative refinement, (2) uniform
  if 'mesh_type' not in params:
    params['mesh_type'] = 'structured' #structured or unstructured
  if 'REF_TYPE' not in params:
    params['REF_TYPE'] = 1   #1) dorfler+tolfref 0) Uniforme 2) Rob refinement
  # refinement until an error iqual to 1+tolref times the cut off error. (tolref=0 default)
  if 'tolref' not in params:
    params['tolref'] = 0.2
  # Percentage (as fraction between 0 and 1) of elements to refine
  if 'REFINE_RATIO' not in params:
    params['REFINE_RATIO'] = 0.25
  # Refinement criterion in ['P','U','mix']
  if 'critref' not in params:
    params['critref'] = 'mix'
    
  # ----------------------------------------------------------------------#
  # Iterative Solver parameters
  # ----------------------------------------------------------------------#
  #Solving using iterative or direct solver
  if 'Solver' not in params:
     params["Solver"] = 'mumps'
  if 'SolverDG' not in params:
      params["SolverDG"] = 'mumpsDG'
     
  #Preconditioner for the Gram matrix (NONE, CHOLESKY,ILU)
  if 'Ptype' not in params:
     params["Ptype"] = 'CHOLESKY'  
  #Preconditioner for the Inverse of G (NONE or CG)
  if 'Gtype' not in params:
     params["Gtype"] = 'NONE'
  #Preconditioner for the Schur Complement (NONE or CHOLESKY)
  if 'Stype' not in params:
     params["Stype"] =  'CHOLESKY'
  #Tolerance to converge the CG for G
  if 'Tol_gram' not in params:
     params["Tol_gram"]    = 1e-12
  #Tolerance to converge the Schur Solver
  if 'Tol_schur' not in params:
     params["Tol_schur"]   = 1e-12
  #Max number of iteration in the iterative solver
  if 'max_iter_cg' not in params:
     params['max_iter_cg'] = 100000
  #Tolerance to converge the Iterative Solver (average previos and current value)
  if 'tol_ave' not in params:
     params['tol_ave']     = 1e-10
  if 'ilu_drop_tol' not in params:
     params['ilu_drop_tol'] = 1E-4
  if 'ilu_fill_factor' not in params:
     params['ilu_fill_factor'] =  10
     
  # ----------------------------------------------------------------------#
  # Damping Newton Solver parameters
  # ----------------------------------------------------------------------#
  if 'tol_newton' not in params:
      params['tol_newton'] = 1e-12 #1e-5
  if 'alpha_d' not in params:
      params['alpha_d'] = 1
  if 'error_delta_e' not in params:
      params['error_delta_e'] = 0.2
  if 'error_delta_u' not in params:
      params['error_delta_u'] = 0.2
  if 'maxiter_d' not in params:
      params['maxiter_d'] = 20000
  if 'omega_d' not in params:
      params['omega_d']=0.5
  if 'factor_d' not in params:
      params['factor_d']=100

  # ----------------------------------------------------------------------#
  # Other parameters
  # ----------------------------------------------------------------------#
  # For plotting purpose
  if 'Exact' not in params:
    params['Exact'] = False
  # For plotting purpose
  if 'Solpng' not in params:
    params['Solpng'] = False
  if 'Solpdf' not in params:
     params['Solpdf'] = False    
    # For plotting purpose
  if 'Solpvd' not in params:
      params['Solpvd'] = False
    # For checkpoint porpouse
  if 'xmlfiles' not in params:
      params['xmlfiles'] = False
    # For plotting purpose  
  if 'Errorpng' not in params:
      params['Errorpng'] = False
      
  if 'meshpng' not in params:
      params['meshpng'] = False
    # Create a folder with some plots of the last solution (last iter) per time
  if 'timepvd' not in params:
      params['timepvd'] = False
  # Name of solution folder
  if 'foldername_solution' not in params:
    params['foldername_solution'] = 'Solution'
  #if 'foldername_con' not in params:
  #  params['foldername_con'] = 'Convergence'
  # Degree of the Expressions approximation
  if 'Degree' not in params:
    params['Degree'] = 8
  # SIP: epsilon=1 or NIP: epsilon=-1 (Symmetric or Non Symmetric formulation)
  if 'epsilon' not in params:
    params['epsilon'] = -1
  if 'eta_adv' not in params:
    params['eta_adv'] = 1. # advection penalization (1 upwinding 0 CF)
  if 'tol' not in params:
    params['tol'] = 1E-14
  if 'vect' not in params:
    params['vect'] = 0
  if 'Constant_tag_BC' not in params:
    params['Constant_tag_BC']=False
  if 'xminplot' not in params:
    params['xminplot']=False
  if 'yminplot' not in params:
    params['yminplot']=False  
  if 'MSupdate' not in params:
    params['MSupdate']= False
  if 'DGprojection' not in params:
    params['DGprojection'] = True
  
  if 'importxml' not in params:
      params['importxml'] = False
  if 'xmaxplot' not in params:
      params['xmaxplot']=False
  if 'ymaxplot' not in params:
      params['ymaxplot']=False    
  if 'linewidth' not in params:
    params['linewidth']  = False
  if 'legendplot' not in params:
    params['legendplot'] = True
    
  if 'ordenplot' not in params:
    params['ordenplot']  = False

      
  if 'q_deg' not in params:
    params['q_deg'] = False
  if 'grad_exa' not in params:
    params['grad_exa'] = False    
  if 'post_end' not in params:
      params['post_end'] = False  
   # ----------------------------------------------------------------------#
   # TEMPORAL parameters
   # ----------------------------------------------------------------------# 
  if 'error_n' not in params:
    params['error_n'] = False
  if 'tiniPM' not in params:  
    params['tiniPM'] = 0

  if 'markers' not in params:
    params['markers'] = []
  if 'DG' not in params:
    params['MS2'] = False
  if 'MS' not in params:
    params['MS2'] = False
  if 'MS2' not in params:
    params['MS2'] = False
  if 'MS1' not in params:
    params['MS1'] = False
  if 'RobinBC' not in params:
    params['RobinBC'] = False

  return params