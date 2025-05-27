# 2025 Juan Felipe Giraldo. Licensed under the MIT License.

import numpy as np
from dolfin import * 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import ASFEM_main as fsim
import sympy as spm
from add_default_parameters import addDefaultParameters
from mesh_generator import MeshGenerator,BoundaryCreator   
from file_manager import FileManager

class ErikssonJhonsson(object):
    def __init__(self, args):
        self.lc    = 2
        self.kappa =  args['kappa']
        self.lamb1 = (-1 + np.sqrt(1 - 4 * self.kappa * self.lc)) / (-2 * self.kappa)
        self.lamb2 = (-1 - np.sqrt(1 - 4 * self.kappa * self.lc)) / (-2 * self.kappa)
        self.r1    = (1 + np.sqrt(1 + 4 * np.pi**2 * self.kappa**2)) / (2 * self.kappa)
        self.s1    = (1 - np.sqrt(1 + 4 * np.pi**2 * self.kappa**2)) / (2 * self.kappa)
        self.vecx  = 1
        self.vecy  = 0
        self.beta_vec = Constant(1) * as_vector([self.vecx, self.vecy])
    
    def forcing_term(self, t, mesh,p):       
        return Constant(0)

    def BoundaryCondition(self, t, p):  
        if p['Unsteady']:
            uNexp = '-kappa*((lamb1*exp(lamb1*x[0]) - lamb2*exp(lamb2*x[0]))*exp(-l*t) + (-r1*exp(r1*x[0]) + s1*exp(s1*x[0]))*cos(M_PI*x[1])/(exp(-s1) - exp(-r1)))'
            uDexp = 'exp(-l*t)*(exp(lamb1*x[0])-exp(lamb2*x[0]))+cos(M_PI*x[1])*(exp(s1*x[0])-exp(r1*x[0]))/(exp(-s1)-exp(-r1))'
            uD = Expression(uDexp, t=t, l=self.lc, lamb1=self.lamb1, lamb2=self.lamb2,
                            s1=self.s1, r1=self.r1, degree=p['Degree'])
            uN = Expression(uNexp, t=t, l=self.lc, lamb1=self.lamb1, lamb2=self.lamb2,
                            s1=self.s1, r1=self.r1, kappa=p['kappa_a'], degree=p['Degree'])
        else:
            uDexp = '(exp(r1*(x[0]-1))-exp(s1*(x[0]-1)))/(exp(-r1)-exp(-s1))*sin(pi*x[1])'
            uD = Expression(uDexp, s1=self.s1, r1=self.r1, degree=p['Degree'])
            uN = Constant(0)
        return [uD, uN]

    def RobinBoundaryCondition(self):   
        return [Constant(0), Constant(0)]
    
    def BoundaryTags(self, p): #tags: [gD,gN,gN2,gR]
        if p['Unsteady']:
            return [[2, 3, 4], [1], [], []]
        else:
            return [[1, 2, 3, 4], [], [], []]

    def Initial(self, t, p):      
        Iniexp = 'exp(-l*t)*(exp(lamb1*x[0])-exp(lamb2*x[0]))+cos(M_PI*x[1])*(exp(s1*x[0])-exp(r1*x[0]))/(exp(-s1)-exp(-r1))'
        return Expression(Iniexp, t=t, l=self.lc, lamb1=self.lamb1,
                          lamb2=self.lamb2, s1=self.s1, r1=self.r1, degree=p['Degree'])
    
    def exact_solution(self, t, p):    
        if p['Unsteady']:
            uexp = 'exp(-l*t)*(exp(lamb1*x[0])-exp(lamb2*x[0]))+cos(M_PI*x[1])*(exp(s1*x[0])-exp(r1*x[0]))/(exp(-s1)-exp(-r1))'
            u_exa = Expression(uexp, t=t, l=self.lc, lamb1=self.lamb1,
                               lamb2=self.lamb2, s1=self.s1, r1=self.r1, degree=p['Degree'])
        else:
            uexp = '(exp(r1*(x[0]-1))-exp(s1*(x[0]-1)))/(exp(-r1)-exp(-s1))*sin(pi*x[1])'
            u_exa = Expression(uexp, s1=self.s1, r1=self.r1, degree=p['Degree'])
        grad_exa = Expression(('0', '0'), t=t, degree=p['Degree'])
        return [u_exa, grad_exa]
    
    def beta_vect_function(self, mesh, p):
        return self.beta_vec
    
    def gamma_function(self, mesh, p):
        return 0
       
    def kappa_function(self, mesh, p):
        return Constant(self.kappa)
    
    def Nonlinear(self, u, p):
        return [0, 0, 0, 0]
    
    def DNonlinear(self, u, p):
        return [0, 0, 0]
    
    def Norm_coefficients(self,mesh):
        return [0, self.beta_vec, Constant(self.kappa), 1.0]

class Anisotropic(object): 
    def __init__(self, args):
        self.r_ani   = args['r_ani']
        self.r_ani2   = args['r_ani2']
        self.tao   = args['tao']


    def forcing_term(self,t,mesh,p):       
        expr= '-1.0/2.0*(2*pow(x[0], 2) - 1)*exp(-r_ani2*tao*pow(x[1], 2) - pow(x[0], 2))/(M_PI*sqrt(1/(r_ani2*tao))) - 1.0/2.0*r_ani2*tao*(2*r_ani2*tao*pow(x[1], 2) - 1)*exp(-r_ani2*tao*pow(x[1], 2) - pow(x[0], 2))/(M_PI*r_ani*sqrt(1/(r_ani2*tao)))'
        return Expression(expr, degree=p['Degree'],  r_ani = self.r_ani, r_ani2 = self.r_ani2 , tao = self.tao)
        

    def BoundaryCondition(self,t,p):  
        expr=  '(1.0/4.0)*exp(-r_ani2*tao*pow(x[1], 2) - pow(x[0], 2))/(M_PI*sqrt(1/(r_ani2*tao)))'
        uD = Expression(expr, degree=p['Degree'],  r_ani2 = self.r_ani2, tao = self.tao)
        uN =  Constant(0)
        return [uD,uN] 
    
    def RobinBoundaryCondition(self):   
        return [Constant(0), Constant(0)]
    
    def BoundaryTags(self,p):    
        return [[1], [], [], []]
  
    def exact_solution(self,t,p):     
        expr=  '(1.0/4.0)*exp(-r_ani2*tao*pow(x[1], 2) - pow(x[0], 2))/(M_PI*sqrt(1/(r_ani2*tao)))'
        u_exa = Expression(expr, degree=p['Degree'], r_ani2 = self.r_ani2, tao = self.tao)
        grad_exa =  Expression(('0','0') ,degree=p['Degree']) #Change
        return [u_exa, grad_exa]
    
    def beta_vect_function(self, mesh, p):
        return 0
    
    def gamma_function(self,mesh,p):    
        return 0
    
    def kappa_function(self,mesh,p): #linear 
        x, y = SpatialCoordinate(mesh) 
        return Expression((('1','0'), ('0','1.0/r_ani')),degree=0, domain=mesh,r_ani = self.r_ani)

    def Norm_coefficients(self,mesh):
        x, y = SpatialCoordinate(mesh) 
        kappa = Expression((('1','0'), ('0','1.0/r_ani')),degree=0, domain=mesh,r_ani = self.r_ani)
        return [0, 0, kappa, 1.0]
    
class Heterogeneous(object):
    def __init__(self, args):
        self.u_half  = args['u_half']
        self.kappa1  = args['kappa1']
        self.kappa2  = args['kappa2']

        self.beta_vec = Constant((1.0, 0.0))

    def forcing_term(self, t, mesh, p):
        return Constant(0.0)

    def BoundaryCondition(self, t, p):
        expr = (
            '(x[0]<0.5)'
            '*( (u_half - exp(1/(2*kappa1)) + (1-u_half)*exp(x[0]/kappa1))'
            '/(1 - exp(1/(2*kappa1))) )'
            '+ (x[0]>=0.5)'
            '*( (-u_half*exp(1/(2*kappa2)) + u_half*exp((x[0]-0.5)/kappa2))'
            '/(1 - exp(1/(2*kappa2))) )'
        )
        uD = Expression(
            expr,
            degree=p['Degree'],
            domain=p['mesh'],
            u_half=self.u_half,
            kappa1=self.kappa1,
            kappa2=self.kappa2
        )
        return [uD, Constant(0.0)]

    def BoundaryTags(self, p):
        return [[1], [], [], []]

    def exact_solution(self, t, p):
        u_exa =  Expression('(x[0]<0.5)*((u_half - exp(1/(2*kappa_1)) + (1 - u_half)*exp(x[0]/kappa_1))/(1-exp(1/(2*kappa_1)))) + (x[0]>=0.5)*((-u_half*exp(1/(2*kappa_2)) + u_half*exp((x[0]-0.5)/kappa_2))/(1-exp(1/(2*kappa_2))))', 
                            u_half=self.u_half, kappa_1=self.kappa1, kappa_2=self.kappa2, degree=p['Degree'], domain=p['mesh'])
        grad_exa = Expression(('0','0'), degree=p['Degree'])
        return [u_exa, grad_exa]

    def beta_vect_function(self, mesh, p):
        return self.beta_vec
    
    def gamma_function(self,mesh,p):
        return 0

    def kappa_function(self, mesh, p):
        return Expression(
            (('(x[0]<0.5)*kappa1 + (x[0]>=0.5)*kappa2','0'),
             ('0','1')),
            degree=0,
            kappa1=self.kappa1,
            kappa2=self.kappa2
        )

    def Norm_coefficients(self, mesh):
        return [0, self.beta_vec, self.kappa_function(mesh, None), 1.0]

class Fichera(object):
    def __init__(self, args):
        self.q_deg   = args['q_deg']
        self.q_const = args['q_const']
        self.kappa   = Constant(1.0)

    def forcing_term(self, t, mesh, p):
        QQ = FiniteElement("Quadrature", mesh.ufl_cell(), degree=self.q_deg, quad_scheme='default')
        U1 = FunctionSpace(mesh, QQ)
        return Expression(
            '-(q*(q+1))*pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], q/2. - 1)',
            element=U1.ufl_element(),
            q=self.q_const
        )

    def BoundaryCondition(self, t, p):
        expr = 'pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], q/2.)'
        uD = Expression(
            expr,
            degree=self.q_deg,
            domain=p['mesh'],
            q=self.q_const
        )
        return [uD, Constant(0.0)]

    def BoundaryTags(self, p):
        return [[1], [], [], []]  # Dirichlet on boundary tag 1

    def exact_solution(self, t, p):
        u_exa = Expression('pow(pow(x[0],2.)+pow(x[1],2.)+pow(x[2],2.),q/2.)',degree=self.q_deg,q=self.q_const)

        grad_exa = Expression(('0','0','0'), degree=self.q_deg)
        return [u_exa, grad_exa]

    def kappa_function(self, mesh, p):
        return self.kappa
    
    def beta_vect_function(self,mesh,p):
        return 0

    def gamma_function(self,mesh,p):
        return 0

    def Norm_coefficients(self, mesh):
        return [0, 0, self.kappa, 1.0]    

class Burgers(object):
    def __init__(self, args):
        self.kappa = args['lamb']
        self.vect = args['vect']

    def forcing_term(self,t,mesh,p):
        fexp = p['ft'];
        return Expression(fexp,kappa=self.kappa,degree=p['Degree'])       
  
    def BoundaryCondition(self,t,p):
        p['uexa'] = '0.5*(1-tanh((2*x[0]-x[1]-0.25)/(sqrt(5*kappa))))'
        uD = Expression(p['uexa'],t=t,kappa=self.kappa,degree=p['Degree'])
        uN = Expression('0',t=t,degree=p['Degree'])
        return [uD,uN]  
  
    def Initial(self,t,p):
        p['Initialexp'] = '0.5'
        f1 = Expression(p['Initialexp'],t=t,kappaini=0.1,degree=p['Degree'])
        return f1  
  
    def BoundaryTags(self,p):        #1 left. 2 top. 3 right. 4 bottom if uses mesh creator
        gN_tags  = []
        gN2_tags = []
        gD_tags  = [1,2,3,4]
        gR_tags  = [] 
        return [gD_tags,gN_tags,gN2_tags,gR_tags]
     
    def RobinBoundaryCondition(self):   
        gR_alpha = Constant(0) # Robin BC 
        gR_u     = Constant(0) # Robin BC  
        return [gR_alpha, gR_u]
  
    def exact_solution(self,t,p):   
        p['uexa'] = '0.5*(1-tanh((2*x[0]-x[1]-0.25)/(sqrt(5*kappa))))'
        p['gradexa'] = ('0','0') 
        u_exa =  Expression(p['uexa'],t=t,kappa=self.kappa,degree=p['Degree'])#   Constant(0)
        grad_exa =  Expression(p['gradexa'],t=t,degree=p['Degree']) #Constant((0,0)) 
        return [u_exa, grad_exa]

    def ComputeForcing_term(self,t,p):
        [x1, x2] = spm.symbols('x[0] x[1]')
        kappa = spm.symbols('kappa')
        F1 = 0.5*(1-spm.tanh((2*x1-x2-0.25)/(spm.sqrt(5*kappa))))
        equ  = F1*F1.diff(x1)+F1*F1.diff(x2)- kappa*F1.diff(x1,2) - kappa*F1.diff(x2,2)
        val = spm.printing.ccode(equ)     
        return val   
    def beta_vect_function(self,mesh,p): #linear
        return 0
    def gamma_function(self,mesh,p): #linear
        return 0
    def kappa_function(self,mesh,p): #linear
       return Constant(self.kappa) 
    def Nonlinear(self, u, p): #reactive term
        F = [0 ,u*p['vect'] ,0,0]             # F(0)u,  F(1)u',  F(2)u''
        return F
    def Nonlinear2(self, u, p): #reactive term
        return  0             # F(0)u,  F(1)u',  F(2)u''
    def DNonlinear(self, u, p): #reactive term
        F = [dot(grad(u),p['vect']), u*p['vect'] , 0]  # DF(0)u, DF(1)u', DF(2)u''    
        return F
    def DNonlinear2(self, u, p): #- alterntative b_upw formulation for (beta*u,grad(v))
        return 0     
    def Norm_coefficients(self, mesh): #anhadir las contribuciones lineales aca tambien
        return [1.0,self.vect,Constant(self.kappa) ,1.0]  # DF(0)u, DF(1)u', DF(2)u'',max(DF(1)): binf      
 
class solvepoissonFEM():
    def solve(self,nx, ny):
        mesh = UnitSquareMesh(nx, ny)
        V = FunctionSpace(mesh, "P", 1)
        u_D = Constant(0.0)
        bc = DirichletBC(V, u_D, "on_boundary")
        f = Expression("10*sin(pi*x[0])*sin(pi*x[1])", degree=2)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(u), grad(v)) * dx
        L = f * v * dx
        u_sol = Function(V)
        solve(a == L, u_sol, bc)    
        return u_sol
 
def test_poisson_FEM_solution():
    u = solvepoissonFEM().solve(10,10)
    norm_u = norm(u, 'L2')
    print("Computed L2 norm of u =", norm_u)
    # reference_value = 0.247140
    # self.assertAlmostEqual(norm_u, reference_value, delta=0.01, 
    #                        msg="L2 norm of solution deviates from reference.")
    return norm_u
def test_steady_Burgers_vms():
    p = addDefaultParameters({})
    steady_updates = {
        'Case':      'Burgers-vms',
        'Nonlinear': True,
        'Nx':        6,
        'Ny':        6,
        'MAX_REF':   5,
        'dofmax':    6e6,
        'pdegree':   1,
        'DG':        True,
        'MS':        True,
        'MS1':       True,
        'Exact'  :   True,
        'Solpng'  :  True, 
        'Solpvd'  :  True,  
        'meshpng' :  True,
        'post_end':  True
    }
    p.update(steady_updates)
    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = 0, 0, 1, 1
    p['pmain'] = 'TestCaseResults_personal/' + p['Case']
    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    lamb = 1e-3
    p['vect']= as_vector([1.0,1.0])
    problem = Burgers({'lamb':lamb, 'vect':p['vect']})
    p['ft'] = problem.ComputeForcing_term(0,p['mesh'])
    FileManager(p).setup_folders()
    Nt = fsim.runSimulation(problem, p)
    error = errornorm(Nt['uexa'], Nt['Usol'], norm_type="L2")
    return error

def test_steady_ErikssonJhonsson():
    p = addDefaultParameters({})
    steady_updates = {
        'Case': 'EJCaseStationary-vms',
        'Unsteady': False,
        'kappa_a': 1e-2,
        'REF_TYPE': 2,
        'MAX_REF': 5,
        'DG': True,
        'MS': True,
        'MS1': True,
        'MSupdate': True,
        'dim': 2,
        'Nx': 4,
        'Ny': 4,
        'dofmax': 2000e3,
        'Solver': 'mumps',
        'SolverDG': 'mumpsDG',
        'Exact': True,
        'Solpng'  :  True, 
        'Solpvd'  :  True,  
        'meshpng' :  True,
        'post_end' : True

    }
    p.update(steady_updates)
    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = 0, 0, 1, 1   
    p['pmain'] = 'TestCaseResults_personal/' + p['Case'] #+ folderset
    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    DicEriksson = {'kappa': p['kappa_a']}
    problem = ErikssonJhonsson(DicEriksson)
    FileManager(p).setup_folders()

    FileManager(p).setup_folders()
    Nt = fsim.runSimulation(problem, p)
    error = errornorm(Nt['uexa'], Nt['Usol'], norm_type="L2")
    return error

def test_unsteady_ErikssonJhonssonBE():
    p = addDefaultParameters({})
    transient_updates = {
        'Case': 'EJCaseTransientBE-vms',
        'Unsteady': True,
        'TIM-PM': 'BE',
        'kappa_a': 1e-2,
        'REF_TYPE': 2,
        'MAX_REF': 4,
        'tfinalPM': 0.1,
        'dtlist': [0.025],
        'ts_ini':0,
        'DG': True,
        'MS': True,
        'MS1': True,
        'MSupdate': True,
        'dim': 2,
        'Nx': 4,
        'Ny': 4,
        'dofmax': 2000e3,
        'Exact': True,
        'Solpng'  :  True, 
        'Solpvd'  :  True,  
        'meshpng' :  True,
        'post_end':  True

    }
    p.update(transient_updates)
    folderset = ('/P' + str(p["pdegree"]) +'-k' + str(p['kappa_a']) + 'upto' + str(p['MAX_REF'])+
                 p['TIM-PM'] + 'dt' + str(p['dtlist'][0]) )

    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = -1, -0.5, 0, 0.5
    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    DicEriksson = {'kappa': p['kappa_a']}
    problem = ErikssonJhonsson(DicEriksson)
    p['pmain'] = 'TestCaseResults_personal/' + p['Case'] + folderset
    T= p['tfinalPM']
    for dt in p['dtlist']:
        p['dt-PM'] = dt
        FileManager(p).setup_folders()
        Nt = fsim.runSimulation(problem, p)
        error = errornorm(Nt['uexa'], Nt['Usol'], norm_type="L2")
    return error

def test_unsteady_ErikssonJhonssonBDF2():
    p = addDefaultParameters({})
    transient_updates = {
        'Case': 'EJCaseTransientBDF2-vms',
        'Unsteady': True,
        'TIM-PM': 'BDF2',
        'kappa_a': 1e-2,
        'REF_TYPE': 2,
        'MAX_REF': 5,
        'tfinalPM': 0.1,
        'dtlist': [0.025],
        'DG': True,
        'MS': True,
        'MS1': True,
        'MSupdate': True,
        'linewidth': 1,
        'dim': 2,
        'Nx': 4,
        'Ny': 4,
        'dofmax': 2000e3,
        'Exact': True,
        'DG': True,
        'Solpng'  :  True, 
        'Solpvd'  :  True,  
        'meshpng' :  True,
        'post_end':  True,
    }
    p.update(transient_updates)
    folderset = ('/P' + str(p["pdegree"])+'-k'+str(p['kappa_a'])+'upto'+str(p['MAX_REF'])+p['TIM-PM']+'dt' + str(p['dtlist'][0]))
    p['pmain'] = 'TestCaseResults_personal/' + p['Case'] + folderset
    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = -1, -0.5, 0, 0.5
    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    DicEriksson = {'kappa': p['kappa_a']}
    problem = ErikssonJhonsson(DicEriksson)
    for dt in p['dtlist']:
        p['dt-PM'] = dt
        FileManager(p).setup_folders()
        Nt = fsim.runSimulation(problem, p)
        error = errornorm(Nt['uexa'], Nt['Usol'], norm_type="L2")
    return error

def test_steady_Anisotropic_vms():
    p = addDefaultParameters({})
    p['pdegree']=2
    
    steady_updates = {
        'REF_TYPE':       2,
        'MAX_REF':        6,
        'pdegree':        2,
        'REFINE_RATIO':   0.25,
        'Degree':         8,
        'r_ani':          1e4,
        'tao':            1e-3,
        'mesh_type':      'structured',
        'dofmax':         1e6,           
        'Ptrial':         p['pdegree'],
        'Ptest':          p['pdegree'],
        'Case':           'Anistropic-vms',
        'Constant_tag_BC': True,
        'dim':            2,
        'Nx':             4,
        'Ny':             4,
        'Exact':          True,
        'DG':             True,
        'MS':             True,
        'MS1':            True,
        'Solpng'  :       True, 
        'Solpvd'  :       True,  
        'meshpng' :       True,
        'post_end':       True

    }
        
    p.update(steady_updates)
    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = -1, -0.5, 1, 0.5

    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    p['pmain'] = 'TestCaseResults_personal/' + p['Case']
    p['eta_0']=1.0

    problem =  Anisotropic({'r_ani':p['r_ani'], 'r_ani2': p['r_ani'], 'tao':p['tao']})
    FileManager(p).setup_folders()

    Nt=fsim.runSimulation(problem,p)
    error = errornorm(Nt['uexa'], Nt['Usol'], norm_type="L2")
    return error

def test_steady_Heterogeneous_vms():
    p = addDefaultParameters({})
    steady_updates = {
        'Case':            'Heterogeneous-vms',
        'Nx':              4,
        'Ny':              4,
        'MAX_REF':         3,
        'dofmax':          2e6,
        'REF_TYPE':        0,
        'Unsteady':       False,
        'Nonlinear':      False,
        'Constant_tag_BC': True,
        'ktensor':         True,
        'SolverDG':        'mumpsDG',
        'Exact':           True,
        'Solpng':          True,
        'Solpvd':          True,
        'DG':              True,
        'MS':              True,
        'MS1':             True,
        'Solpng'  :        True, 
        'Solpvd'  :        True,  
        'meshpng' :        True,
        'post_end':        True
    }
    p.update(steady_updates)

    p['pmain'] = 'TestCaseResults_personal/' + p['Case']
    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = 0, 0, 1, 1
    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    kappa1 = 0.1
    kappa2 = 1.0
    e1 = np.exp(1/(2*kappa1))
    e2 = np.exp(1/(2*kappa2))
    u_half = (e1/(1-e1)) * (e1/(1-e1) + 1/(1-e2))**(-1)
    problem = Heterogeneous({
        'u_half': u_half,
        'kappa1': kappa1,
        'kappa2': kappa2
    })
    FileManager(p).setup_folders()
    Nt = fsim.runSimulation(problem, p)
    u_exa, _ = problem.exact_solution(0, p)
    error = errornorm(u_exa, Nt['Usol'], norm_type="L2")
    return error

def test_steady_Fichera_vms():
    p = addDefaultParameters({})
    steady_updates = {
        'Case':         'Fichera-vms',
        'MAX_REF':      3,
        'REF_TYPE':     2,
        'REFINE_RATIO': 0.125,
        'dim':          3,
        'Nx':           2,
        'Ny':           2,
        'Degree':       2,
        'mesh_type':    'L-shape-3D',
        'Constant_tag_BC': True,
        'Solver':       'Iterative',
        'Exact':        True,
        'legendplot':   False,
        'DG':           True,
        'MS':           True,
        'MS1':          True,
        'Solpng'  :     True, 
        'Solpvd'  :     True,  
        'meshpng' :     True,
        'post_end':     True
    }
    p.update(steady_updates)

    p['q_const'] = 1.0/10.0
    p['q_deg']   = p['Ptrial'] + 4
    p['pmain'] = 'TestCaseResults_personal/' + p['Case']
    p['Linfx'], p['Linfy'], p['Lsupx'], p['Lsupy'] = 0, 0, 1, 1
    p['mesh'] = MeshGenerator(p).Create_mesh()
    p['boundaries'] = BoundaryCreator(p,p['mesh']).create_boundaries()
    problem = Fichera({
        'q_deg': p['q_deg'],
        'q_const': p['q_const']
    })
    FileManager(p).setup_folders()
    Nt = fsim.runSimulation(problem, p)
    u_exa, _ = problem.exact_solution(0, p)
    error = errornorm(u_exa, Nt['Usol'], norm_type="L2")
    return error
