# 2025 Juan Felipe Giraldo. Licensed under the MIT License.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:55:45 2025
"""
import fenics
from dolfin import * 
import numpy as np

class DGForms:
    def __init__(self, n, h, args):
        """
        ArgsDic: Dictionary containing necessary parameters. Expected keys include:
          "dx": volume measure,
          "dS": interior facet measure,
          "ds": boundary measure,
          "dsD": Dirichlet measure,
          "eta": stabilization parameter,
          "psuperc": exponent for cell diameter,
          "epsilon": penalty scaling factor,
          "gammac": reaction (or consistency) coefficient,
          "kappa": diffusivity,
          "beta_vec": advection vector,
          and optionally keys for norm definitions such as "gamma_norm", "kappa_norm", "beta_norm",
          "beta_inf", "eta_adv", and for Robin: "dsR", "gR_alpha".
        """
        self.args = args
        # --- measures (already UFL Measure objects) ---
        self.dx   = args["dx"]
        self.dsD   = args['dsD']
        self.dsR   = args['dsR']
        self.dsN   = args['dsN']
        self.dsN2  = args['dsN2']
        self.dS    = args["dS"]
        self.ds    = args["ds"]
        # --- boundary data ---
        self.gD    = args['gD']
        self.gN    = args['gN']
        # --- stabilization & penalty ---
        self.eta_adv = args['eta_adv']
        self.epsilon = args['epsilon']
        self.eta_adv = args['eta_adv']
        self.eta = args['eta']/(h**args['psuperc']);
        # --- norm-weights ---
        self.gamma_norm = args['gamma_norm']
        self.kappa_norm = args['kappa_norm']
        self.beta_norm = args['beta_norm']
        self.beta_inf = args['beta_inf']
        # --- constant coefficients ---
        self.gamma =    args['gammac']
        self.kappa =    args['kappa'];
        self.beta_vec = args['beta_vec'];
        # --- flags & nonlinear term object (not UFL) ---
        self.Unsteady  = args['Unsteady']
        self.Nonlinear = args['Nonlinear']
        self.NLterm    = args['NLterm']
        
        self.n = n
        self.h = h
        
    def gamma_K_b(self,kappa):
        gamma_K_b = dot(self.n, kappa*self.n)
        return gamma_K_b
        
    def gamma_K_i(self,kappa):
        delta_Kn_plus = dot(self.n('+'), kappa('+')*self.n('+'))
        delta_Kn_min = dot(self.n('-'), kappa('-')*self.n('-'))
    
        gamma_K_i = delta_Kn_plus*delta_Kn_min/(delta_Kn_plus + delta_Kn_min)
        return gamma_K_i
    
    def avgK(self,kappa, v):
        delta_Kn_plus = dot(self.n('+'), kappa('+')*self.n('+'))
        delta_Kn_min = dot(self.n('-'), kappa('-')*self.n('-'))
    
        omega_min = delta_Kn_plus/(delta_Kn_plus + delta_Kn_min)
        omega_plus = delta_Kn_min/(delta_Kn_plus + delta_Kn_min)    
        val = omega_min*(kappa*grad(v))('-') + omega_plus*(kappa*grad(v))('+')
        return val
    
    def gh(self,e,w,grade,gradw,c): 
         val = 0
         C2 = 1
         if self.Unsteady:
             val =   self.args['C1']*self.ghV0(e,w,c)   
             C2  =   self.args['C2']
         val +=  C2*self.gh_dar(e,w,grade,gradw,c)    
         return val
    
    def ghV0(self,e,w,c):
          return inner(e,w*c)*self.dx
    
    #space-dependent norms
    def gh_dar(self,e,w,grade,gradw,c):
     #norm sip
        val =0
        if self.gamma_norm:
           val += (self.gamma_norm)*inner(e, w)*c*self.dx 
        if self.kappa_norm: #:
           val += self.gh_sip(e,w,grade,gradw,c)
        if self.beta_norm:
           val += self.gh_upw(e,w,grade,gradw,c)
        if self.kappa_norm == 0 and self.beta_norm == 0:
           raise Exception('you must specify at least one, kappa or beta_vec')  
        return val
    
    def gh_upw(self,e,w,grade,gradw,c):
     #norm upw
        val = 0.5*inner(abs(dot(self.beta_norm,self.n))*e, w*c)*self.ds  #semi norm
        val += self.eta_adv*0.5*inner(abs(dot(self.beta_norm('+'),self.n('+')))*jump(e), jump(w)*avg(c))*self.dS  # upw
        val += self.h/self.beta_inf*inner(dot(self.beta_norm, grade), c*dot(self.beta_norm, gradw))*self.dx  #inf sup
        val += self.beta_inf*inner(e,w*c)*self.dx 
        return val

    def gh_sip(self,e,v,grade,gradv,c):
        kappaN = self.kappa_norm      
        val = inner(kappaN*grade, gradv*c)*self.dx 
        val += inner(self.gamma_K_i(kappaN)*avg(self.eta)*jump(e), jump(v)*avg(c))*self.dS
        val += inner((self.gamma_K_b(kappaN)*self.eta)*e, v*c)*self.ds  #penalty
        return val
  
    # ----------------------------------------------------------------------#
    # DG Nonlinear Bilinear Forms
    # ----------------------------------------------------------------------#
    
    def bh_prima(self,u_k,du,v):   #fully bh primer trilinear form
        val = 0
        C2 = 1.0
        if self.Unsteady:
           val =   self.args['C1']* inner(du,v)*self.dx
           C2  =   self.args['C2']
        val += C2 * self.bh_dar(du,v)
        
        if self.Nonlinear:
           val += C2 * self.bh_dar_DNL(u_k,du,v)        
        return val
    
    def bh(self,u,v):   #fully bh trilinear form
        val = 0
        C2 = 1
        if self.Unsteady:
           val =   self.args['C1']* inner(u,v)*self.dx
           C2  =   self.args['C2']
        val += C2 * self.bh_dar(u,v)
        if self.Nonlinear:
            val += C2 * self.bh_dar_NL(u,v)
    
        return val
    
    def bht(self,v,u):   #fully bh trans trilinear form
        val = 0
        C2 = 1
        if self.Unsteady:
           val =   self.args['C1']* inner(v,u)*self.dx
           C2  =   self.args['C2']
        val += C2 * self.bh_dar(v,u)
        if self.Nonlinear:
            val += C2 * self.bh_dar_NL(v,u)  
        return val
    
    def bh_dar(self,u,v):
        val = 0
        if self.gamma:      val += inner(u, self.gamma*v)*self.dx 
        if self.kappa:      val += self.bh_sip(u,v,self.kappa)
        if self.beta_vec:   val += self.bh_upw(u,v,self.beta_vec)
        if self.kappa== 0 and  self.beta_vec== 0 and self.Nonlinear == 0:
           raise Exception('you must specify at least one, kappa or beta_vec')
        return val
    
    def bh_dar_NL(self,du,v):
        gamma =       self.NLterm.Nonlinear(du,self.args)[0] 
        beta_vec =    self.NLterm.Nonlinear(du,self.args)[1] 
        kappa =       self.NLterm.Nonlinear(du,self.args)[2] 
        val = 0    
        if gamma:     val += inner(v, gamma)*self.dx   #reactive nonlinar term         
        if beta_vec:  val += self.bh_upw(du,v,beta_vec)  #advective nonlinar term
        if kappa:     val += self.bh_sip(du,v,kappa)  #diffusive nonlinar term          
        return val
    
    def bh_dar_DNL(self,u_k,du,v):
        gamma =       self.NLterm.DNonlinear(u_k,self.args)[0] 
        beta_vec =    self.NLterm.DNonlinear(u_k,self.args)[1] 
        kappa =       self.NLterm.DNonlinear(u_k,self.args)[2] 
        val = 0    
        if gamma:     val += inner(v, gamma*du)*self.dx   #reactive nonlinar term         
        if beta_vec:  val += self.bh_upw(du,v,beta_vec)  #advective nonlinar term
        if kappa:     val += self.bh_sip(du,v,kappa)  #diffusive nonlinar term            
        return val
    
    def bh_sip(self,u,v,kappa):
        #SWIP
        val = inner(kappa*grad(u), grad(v))*self.dx 
        val += -inner(dot(self.avgK(kappa,u), self.n('+')), jump(v))*self.dS #consistency
        val += self.epsilon*inner(jump(u), dot(self.avgK(kappa, v), self.n('+')))*self.dS #adjoint consistency
        val += inner(avg(self.eta)*self.gamma_K_i(kappa)*jump(u), jump(v))*self.dS #penalty
        val += -inner(v, dot(kappa*grad(u), self.n))*self.dsD #consistency
        val += self.epsilon*inner(u, dot(kappa*grad(v), self.n))*self.dsD #adjoint consistency
        val += inner(self.eta*self.gamma_K_b(kappa)*u, v)*self.dsD  #penalty
        
        if self.dsR != None:
            val += inner(self.args['gR_alpha']*self.gamma_K_b(kappa)*u,v)*self.dsR  #Robin
        return val
    
    def bh_upw(self,u,v,beta_vec):
        #UPW
        c = Constant(1)
        val = inner(dot(beta_vec,grad(u)), v)*c*self.dx 
        val += 0.5*inner((abs(dot(beta_vec,self.n))-dot(beta_vec,self.n))*v, u)*c*self.ds
        val -= inner(dot(beta_vec('+'),self.n('+'))*avg(v), jump(u))*avg(c)*self.dS
        val += 0.5*inner(abs(dot(beta_vec('+'),self.n('+')))*jump(u), jump(v))*avg(c)*self.dS
        return val
    
    
    # ----------------------------------------------------------------------#
    # DG RHS
    # ----------------------------------------------------------------------#
    def rhs(self,u_k,w,f):  #fully rhs form
        val = 0
        Crhs3  = 1
        if self.Unsteady:
              Crhs1 = self.args["Crhs1"] ; Crhs2 = self.args["Crhs2"] ; Crhs3 = self.args["Crhs3"]
              if Crhs1: val += Crhs1 * inner(self.args["u_n2"],w)*self.dx
              val += Crhs2 * inner(self.args["u_n"],w)*self.dx
        val += Crhs3 * inner(f,w)*self.dx  #Source term
        val +=  Crhs3 * self.rhs_dar(w)
        if self.Nonlinear:
            val += Crhs3 * self.rhs_dar_NL(u_k,w,self.args['f'])
        return val
    
    def rhs_dar(self,w):
        val = 0
        if self.dsN:
          C = Constant(1)  
          val += inner(C*self.gN, w)*self.dsN #Neumann boundary conditions
        if self.dsN2:  
          if self.args['NeumannVar'] == False:
               C = self.args['g2N']
          else:
               vec=as_vector([0,self.args['g2N']])
               C = dot(dot(kappa,self.n),vec) 
          val += inner(C, w)* self.dsN2
    
        if self.args['dsR'] and self.kappa: # to do for Nonlinear (working only for Linears)
          val += inner(self.args['gR_alpha']*self.kappa*self.args['gR_u'], w)*self.dsR #Robin boundary conditions
        if self.kappa:     val += self.rhs_sip(w,self.kappa)  #SIP RHS contribution
        if self.beta_vec:  val += self.rhs_adv(w,self.beta_vec)  #Upwinding RHS contribution
        if self.kappa==0 and self.beta_vec==0 and self.Nonlinear == 0:
           raise Exception('you must specify at least one, kappa or beta_vec')
        return val
    
    def rhs_dar_NL(self,u_k, w,f):  
        kappa = self.NLterm.Nonlinear(u_k,self.args)[2] 
        beta_vec = self.NLterm.Nonlinear(u_k,self.args)[1] 
        val = 0
        if kappa:     val += self.rhs_sip(w,kappa)  #SIP RHS contribution
        if beta_vec:  val += self.rhs_adv(w,beta_vec)  #Upwinding RHS contribution 
        return val

    def rhs_sip(self,w,kappa):
        val   = 0;
        val   = self.epsilon*inner(dot(kappa*grad(w),self.n), self.gD)*self.dsD
        val  += inner(self.gamma_K_b(kappa)*self.eta*w, self.gD)*self.dsD
        return val
    
    def rhs_adv(self,w,beta_vec):
        return self.eta_adv*0.5*inner((abs(dot(beta_vec,self.n))-dot(beta_vec,self.n))*w, self.gD)*self.ds


class OptimalStabilization:
    def __init__(self, mesh, dx, ds, dS):
        self.mesh = mesh
        self.dx   = dx
        self.ds   = ds
        self.dS   = dS

    def compute(self):
        PC = FunctionSpace(self.mesh, "DG", 0)
        c  = TestFunction(PC)
        sta = Function(PC)

        AK = np.array(assemble(2*Constant(1) * avg(c) * self.dS + Constant(1)* c * self.ds))
        VK = np.array(assemble(Constant(1) * c * self.dx))

        sta.vector()[:] = AK / VK
        return sta
    
    def compute_eta(self, params):
        dim  = self.mesh.topology().dim()
        pdeg = params['pdegree']
        return (pdeg + 1) * (pdeg + dim) / dim * self.compute() *params['eta_0']
     