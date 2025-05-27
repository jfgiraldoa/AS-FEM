#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 23:13:55 2025

@author: Juan Giraldo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpltools import annotation
from dolfin import *
from function_space_utils import FunctionSpaceUtils as fsu
from dg_forms import DGForms
from solvers_asfem import Solvers

class Postprocessor:
    """
    Post‐processing routines: saving and plotting solutions and convergence data.

    :param ArgsDic: dict of DG/ASFEM arguments
    :param params:  dict of user parameters for plotting and control flags
    """
    def __init__(self, ArgsDic, params):
        self.ArgsDic = ArgsDic
        self.params  = params


    def postprocess_plots(self,SolDic):
          if self.params['meshpng']: self.plot_mesh(self.ArgsDic['mesh'],'mesh',self.ArgsDic['iterR'],'png',self.ArgsDic['folsolve'])
          if self.params['Solpng']: 
              self.plota(SolDic["Usol"],'solCG',self.ArgsDic['iterR'],'png',self.ArgsDic['folsolve'])
          if self.params['Errorpng']: 
              self.plota(SolDic["Esol"],'E_DG',self.ArgsDic['iterR'],'png',self.ArgsDic['folsolve'])
    
          if self.params['Solpdf']: 
              self.plota(SolDic["Usol"],'solCG',self.ArgsDic['iterR'],'pdf-sol',self.ArgsDic['folsolve'])   
              if self.params['DG']: self.plota(SolDic["UsolDG"],'solDG',self.ArgsDic['iterR'],'pdf-sol',self.ArgsDic['folsolve']) 
                  
          if self.params['Solpvd']: 
              self.plota(SolDic["Usol"],'solCG'+str(self.ArgsDic['iterR']),self.ArgsDic['iterR'],'pvd',self.ArgsDic['folsolve'])
              if self.params['DG']: self.plota(SolDic["UsolDG"],'solDG',self.ArgsDic['iterR'],'pvd',self.ArgsDic['folsolve']) 
              
          if self.params['DG'] and self.params['Solpng']: self.plota(SolDic["UsolDG"],'solDG',self.ArgsDic['iterR'],'png',self.ArgsDic['folsolve']) 
          if self.params['MS']: 
             if self.params['Solpdf']:
                 self.plota(SolDic["u_primaDG"],'u_primaDG',self.ArgsDic['iterR'],'pdf',self.ArgsDic['folsolve'])  
             if self.params['Solpng']:
                  self.plota(SolDic["u_primaDG"],'u_primaDG',self.ArgsDic['iterR'],'png',self.ArgsDic['folsolve'])
             if self.params['Solpng']:
                 self.plota(SolDic['UsolDGnew'],'u_newDG',self.ArgsDic['iterR'],'png',self.ArgsDic['folsolve'])
             if self.params['Solpvd']:
                 self.plota(SolDic["u_primaDG"],'u_primaDG',self.ArgsDic['iterR'],'pvd',self.ArgsDic['folsolve'])
             if self.params['Solpvd']:
                 self.plota(SolDic['UsolDGnew'],'u_newDG',self.ArgsDic['iterR'],'pvd',self.ArgsDic['folsolve'])
    
          return 0

    def save_object(self, var, name_ref, iteration, folder, form):
        if self.ArgsDic.get('Unsteady', False):
            namesol = f"{name_ref}-ts{self.ArgsDic['Time_step']}"
        else:
            namesol = name_ref
        self.plot_a(var, namesol, iteration, form, folder)

    def plot1D(self, f):
        """
        Plot a 1D function via its vertex values, or fallback to dolfin.plot.
        """
        mesh = f.function_space().mesh()
        if mesh.topology().dim() == 1:
            C = f.compute_vertex_values(mesh)
            X = mesh.coordinates()[:,0]
            idx = np.argsort(X)
            plt.plot(X[idx], C[idx], linewidth=1, color='grey')
        else:
            plot(f)


    def plota(self, var, name, itera, form, folder):
        """
        Internal helper for various plot/save formats.
        """
        os.makedirs(folder, exist_ok=True)
        # determine output filename
        if form == 'pvd':
            fname = os.path.join(folder, f"{name}.pvd")
            var.rename('var', 'var')
            vtkfile = File(fname)
            vtkfile << (var, itera)
            return
        if form == 'xml':
            fname = os.path.join(folder, f"{name}.xml")
            xmlfile = File(fname)
            xmlfile << var
            return

        # other formats include iteration in filename
        fname = os.path.join(folder, f"{name}{itera}.{form}")
        if form == 'pdf-sol':
            fig, ax1 = plt.subplots()
            cf = plot(var)
            plt.colorbar(cf)
            plot(var.function_space().mesh(), linewidth=0.1)
            ax1.ticklabel_format(useOffset=True, style='plain')
            plt.xticks([]); plt.yticks([])
            fig.savefig(fname, dpi=500, format='pdf')
            plt.close()
        elif form == 'pdf':
            fig, ax1 = plt.subplots()
            cf = plot(var)
            plt.colorbar(cf)
            ax1.ticklabel_format(useOffset=True, style='plain')
            plt.xticks([]); plt.yticks([])
            fig.savefig(fname, dpi=500, format='pdf')
            plt.close()
        else:
            if self.params['dim'] < 3:
                fig, ax1 = plt.subplots()
                if self.params['dim'] == 1:
                    self.plot1D(var)
                elif self.params['dim'] == 2:
                    plot(var)

                ax1.ticklabel_format(useOffset=True, style='plain')
                fig.savefig(fname, dpi=500, format=form)
                plt.close()



    def plot_mesh(self, var, name, iteration, form, folder):
        """
        Plot mesh only.
        """
        if self.params['dim'] < 3:
            os.makedirs(folder, exist_ok=True)
            fig = plt.figure()
            plot(var)
            fname = os.path.join(folder, f"{name}{iteration}.{form}")
            fig.savefig(fname, dpi=500, format=form); plt.close()

    def plotexa(self,uexa,V,iterR,folsolve):
        Usol_exa = interpolate(uexa,V)
        if self.params['Solpng']: self.plota(Usol_exa,'solU-exa',iterR,'png',folsolve)
        if self.params['Solpvd']: self.plota(Usol_exa,'solU-exa',iterR,'pvd',folsolve)    

    def plot_multilog(self,varx,vary,name,marker,colordic,form,crit,varref,time,DiscType):
       fig,ax1= plt.subplots()
       
       if self.params['REF_TYPE']:
           reflabel = '-ada'
       else:
           reflabel = '-uni'
       
       if varref == 'Converg_L2':
          varname = 'Converg_L2_p'+str(self.params['pdegree'])+reflabel
    
          order=self.params['pdegree']+1
          ylabel = '$L_2$-norm'
          
       elif   varref == 'Converg_Vh' :
          varname = 'Converg_Vh_p'+str(self.params['pdegree'])+reflabel
          order=self.params['pdegree']
          ylabel = '$V_h$-norm'
               
       elif  varref == 'Converg_Vh-noerr':
           varname = 'Converg_Vh_p-NE'+str(self.params['pdegree'])+reflabel
           order=self.params['pdegree']
           ylabel = '$V_h$-norm'    
             
       elif varref == 'Converg_L2dt' or varref == 'Converg_Vhdt'  : #change Vhdt
           varname = varref
           if self.params['TIM-PM'] == 'BE':
             order = 1
           elif params['TIM-PM'] == 'BDF2':
             order = 2  
           ylabel = 0
       else:
           order=0
           ylabel = 0
       if self.params['ordenplot']: order = self.params['ordenplot']    
    
       for i in vary:#range (len(vary)):
         if crit == 'log-log':
           line,= ax1.loglog(varx,vary[i],marker[i],color=colordic[i],markersize=4)#,color="black")
           if self.params['linewidth'] :line.set_linewidth(self.params['linewidth'])
         elif crit == 'logy':
           line,= ax1.semilogy(varx,vary[i],marker[i],color=colordic[i],markersize=4)#, color="black")
         elif crit == 'logx':
           line,= ax1.semilogx(varx,vary[i],marker[i],color=colordic[i],markersize=4)#,color="black")
    
         line.set_label(name[i])
    
       if DiscType == 'space':
           dire = 1
           booldir = 'False'      
           if self.params['xminplot']  : 
             plt.xlim(self.params['xminplot'])
             if self.params['xmaxplot']  : plt.xlim([self.params['xminplot'],self.params['xmaxplot']])
    
           if self.params['yminplot']  : 
             plt.ylim(self.params['yminplot'])
             if params['ymaxplot']  : plt.ylim([self.params['yminplot'],self.params['ymaxplot']])
         
           if self.params['legendplot']: plt.legend(frameon=False)
    
    
           if self.params['REF_TYPE']==0: plt.xlabel("$h$"); plt.gca().invert_xaxis()
           else:  plt.xlabel("$DoF^{-1/"+str(self.params['dim'])+"}$");plt.gca().invert_xaxis()
           
           if ylabel: plt.ylabel(ylabel)
    
       else:
           plt.xlabel("$\Delta t$");
           dire = 1
           booldir = 'True'
       if order:
         for i in vary:#range (len(vary)):
           posx = varx[-1]
           posy=vary[i][-1]#tiempo eval, variable, dt, posicion
           s_scale=0.07
           annotation.slope_marker((posx*0.95, posy*0.95), (dire*1*order), ax=ax1,size_frac=s_scale,invert=booldir,
                                poly_kwargs={'facecolor': 'white','edgecolor': 'black'})
    
       if DiscType == 'space' and self.params['Unsteady']: 
               ref = 't-'+ str(time)
       else: 
               ref = ''
       
       if DiscType == 'space' and self.params['MAX_REF']>1:  
           folder = self.params['foldername_con']
       else:
               folder = self.params['foldername_con_time']
       
       TrialTypeDir = os.path.join(folder)
       if not os.path.isdir(TrialTypeDir): os.makedirs(TrialTypeDir)
       data_filename = os.path.join(TrialTypeDir,varname+ref+'.'+form)
       fig.savefig(data_filename, dpi= 500, format=form)
    
       plt.close()       
 
    def postprocess_convergence_space(self,NormDic,time_step):

            datadic2 = {'norm_eVh':NormDic["E"]} 

            namedic = {
                    'norm_eVh' :    "$||e_h||_{Vh}$",
                    'norm_eGOA' :   "$||estar_h||_{Vh}$",
                    'norm_res0':    "$||u-u_h||_{L2}$",
                    'norm_res1':    "$||u-uDG_h||_{L2}$",
                    'norm_res2':    "$||u-(u_h+u_hDG')||_{L2}$",
                    'norm_res3':    "$||u-u_h||_{Vh}$",
                    'norm_res4':    "$||u-uDG_h||_{Vh}$",
                    'norm_res5':    "$||u-(u_h+u_hDG')||_{Vh}$",
                    'norm_res6':    "$||u-(u_h+u_hDG2')||_{Vh}$",
                    'norm_res7':    "$||u-(u_h+u_hDG2')||_{L2}$",
                    'norm_res8':    "$||u-(u_h+u_hDG1')||_{Vh}$",
                    'norm_res9':    "$||u-(u_h+u_hDG1')||_{L2}$",

                      }
            markerdic = {
                    'norm_eVh':    '+-',
                    'norm_eGOA':    '*-',
                    'norm_res0':    '--',
                    'norm_res1':    '*-',
                    'norm_res2':    '*--',
                    'norm_res3':    '-',
                    'norm_res4':    '-',
                    'norm_res5':    '*-',
                    'norm_res6':    '*--',
                    'norm_res7':    '*--',
                    'norm_res8':    '--',
                    'norm_res9':    '--',
                      }
            colordic = {
                  'norm_eVh' :  "red",
                  'norm_res0':  'blue',
                  'norm_eGOA':  'yellow',
                  'norm_res1':  'black',
                  'norm_res2':  'green',
                  'norm_res3':  'blue',
                  'norm_res4':  'black',
                  'norm_res5' : "green",
                  'norm_res6' : "orange",
                  'norm_res7' : "orange",
                  'norm_res8' : "brown",
                  'norm_res9' : "brown",
                      }

            if self.params['Exact']:
                datadic2.update({'norm_res3':NormDic["VherrCG"]})
                datadic3 = {'norm_res3':NormDic["VherrCG"]}
                datadic1 = {'norm_res0':NormDic["L2errCG"]} # normL2
                
                if self.params['DG']:
                    datadic3.update({'norm_res4':NormDic["VherrDG"]})
                    datadic2.update({'norm_res4':NormDic["VherrDG"]})
                    datadic1.update({'norm_res1':NormDic["L2errDG"]})
                if self.params['MS']:
                    datadic3.update({'norm_res5':NormDic["VherrnewDG"]})
                    datadic2.update({'norm_res5':NormDic["VherrnewDG"]})
                    datadic1.update({'norm_res2':NormDic["L2errnewDG"]})    
                if self.params['MS1']:
                     datadic3.update({'norm_res8':NormDic["VherrnewDG1"]})
                     datadic2.update({'norm_res8':NormDic["VherrnewDG1"]})
                     datadic1.update({'norm_res9':NormDic["L2errnewDG1"]})                    
                if self.params['MS2']:
                    datadic3.update({'norm_res6':NormDic["VherrnewDG2"]})
                    datadic2.update({'norm_res6':NormDic["VherrnewDG2"]})
                    datadic1.update({'norm_res7':NormDic["L2errnewDG2"]})   
      
                self.plot_multilog(NormDic["Dofsvect"],datadic1,namedic,markerdic,colordic,'png','log-log','Converg_L2',time_step,'space')
                self.plot_multilog(NormDic["Dofsvect"],datadic3,namedic,markerdic,colordic,'png','log-log','Converg_Vh-noerr',time_step,'space')

            self.plot_multilog(NormDic["Dofsvect"],datadic2,namedic,markerdic,colordic,'png','log-log','Converg_Vh',time_step,'space')

            return 0
        
    def postprocess_NLresiduos(self,SolDic):
            V  = self.ArgsDic['Hh'][2];
            mesh = self.ArgsDic['mesh']
            e,u,v,w = fsu.get_TestTrial(V,self.ArgsDic)
            n = FacetNormal(mesh)
            h = CellDiameter(mesh)
            dg = DGForms(n,h,self.ArgsDic)

            lh1  = dg.rhs(SolDic["Usol"],w,self.ArgsDic['f'])
            bh1  = dg.bh(SolDic["Usol"],w)
            gh1  = dg.gh(SolDic["Esol"],w,grad(SolDic["Esol"]),grad(w),Constant(1))
            bhp1 = dg.bh_prima(SolDic["Usol"],v,SolDic["Esol"])

            # TOTAL CONVERGENCE
            residual_fin = Function(V)
            residual_fin_2 = Function(V)
            residual_fin_u = Function(V)
            residual_fin_e = Function(V)
            residual_fin.vector()[:] = np.array(assemble(lh1 - bh1 - gh1 - bhp1))

            residual_fin_2.vector()[:] = np.array(assemble(lh1 - bh1))
            residual_fin_e.vector()[:] = np.array(assemble(lh1 - bh1 - gh1))
            residual_fin_u.vector()[:] = np.array(assemble(-bhp1))

            residual_norm_fin = np.linalg.norm(residual_fin.vector()[:])
            residual_norm_fin_2 = np.linalg.norm(residual_fin_2.vector()[:])
            residual_norm_u = np.linalg.norm(residual_fin_u.vector()[:])
            residual_norm_e = np.linalg.norm(residual_fin_e.vector()[:])
            print('residual norm = ', residual_norm_fin)
            print('L-BU norm = ', residual_norm_fin_2)
            print('residual norm u = ', residual_norm_u)
            print('residual norm e = ', residual_norm_e)
            if self.params['Case']=='Bratu':
                print('u_cg(0.5,0.5) = ', SolDic["Usol"](0.5,0.5))
                if self.params['DG']: print('u_dg(0.5,0.5) = ', SolDic["UsolDG"](0.5,0.5))

            return residual_norm_fin,residual_norm_fin_2
        
        
    def postprocess_norms(self,SolDic,NormDic):
        NormDic["E"].append(SolDic["E"]);
        print('E'+':',SolDic["E"])
        mesh = self.ArgsDic['mesh']
        n = FacetNormal(self.ArgsDic['mesh'])
        h = CellDiameter(self.ArgsDic['mesh'])
        dx = self.ArgsDic['dx']
        dg = DGForms(n,h,self.ArgsDic)

        if self.params['Exact']:
            uexa   = SolDic["uexa"]

            Vd = SolDic["Esol"].function_space()
            Ud = SolDic["Usol"].function_space()

            ud = TrialFunction(Vd)
            wd = TestFunction(Vd)

            n = FacetNormal(self.ArgsDic['mesh'])
            h = CellDiameter(self.ArgsDic['mesh'])
          
            CGerror = Function(Vd)
            usolCG = SolDic['UsolCG-dis']

            if self.params['grad_exa']:  
                uexaDG = fsu.projection_exa(uexa,self.ArgsDic['gradexa'],Vd,'H1',0,self.ArgsDic)
                uexaCG = fsu.projection_exa(uexa,self.ArgsDic['gradexa'],Ud,'H1',0,self.ArgsDic)

            else:
                uexaDG = project(uexa,Vd)  
                uexaCG = project(uexa,Ud)

            CGerror.vector()[:] = uexaDG.vector()[:] - usolCG.vector()[:]
            L2errCG = sqrt(assemble(inner(CGerror,CGerror)*dx))
            VherrCG   =  sqrt(assemble(dg.gh(CGerror,CGerror,grad(CGerror),grad(CGerror),Constant(1)))) #+ bhrhs

            NormDic["L2errCG"].append(L2errCG)  ;
            NormDic["VherrCG"].append(VherrCG) ;
            NormDic['hmin'].append(mesh.hmin())


            print('l2 h norm = ', L2errCG)
            print('Vh h norm = ', VherrCG)

            if self.params['DG']:
                usolDG = SolDic["UsolDG"]
                DGerror = Function(Vd)
                DGerror.vector()[:] = uexaDG.vector()[:] - usolDG.vector()[:]
                L2errDG  = sqrt(assemble(inner(DGerror,DGerror)*dx))
                VherrDG   =  sqrt(assemble(dg.gh(DGerror,DGerror,grad(DGerror),grad(DGerror),Constant(1)))) #+ bhrhs
                NormDic["L2errDG"].append(L2errDG) ;
                NormDic["VherrDG"].append(VherrDG) ;

                print('l2 nrDG norm = ', L2errDG)
                print('Vh nrDG norm = ', VherrDG)


            if self.params['MS']:
                u_newDGvec = SolDic['UsolDGnew']
                DGnewerror = Function(Vd)
                DGnewerror.vector()[:] = uexaDG.vector()[:] - u_newDGvec.vector()[:]
               
                VherrnewDG   =  sqrt(assemble(dg.gh(DGnewerror,DGnewerror,grad(DGnewerror),grad(DGnewerror),Constant(1)))) #+ bhrhs1+bhrhs1
                L2errnewDG   =  sqrt(assemble(inner(DGnewerror,DGnewerror)*dx))
                NormDic["L2errnewDG"].append(L2errnewDG) ;
                NormDic["VherrnewDG"].append(VherrnewDG) ;
                
                print('l2 nrNewDG norm = ', L2errnewDG)
                print('Vh nrNewDG norm = ', VherrnewDG)
               
            if self.params['MS1']:
               u_newDGvec1 = SolDic['UsolDGnew1']
               DGnewerror1 = Function(Vd)
                  
               DGnewerror1.vector()[:] = uexaDG.vector()[:] - u_newDGvec1.vector()[:]
                  
               VherrnewDG1   =  sqrt(assemble(dg.gh(DGnewerror1,DGnewerror1,grad(DGnewerror1),grad(DGnewerror1),Constant(1)))) #+ bhrhs1
               L2errnewDG1   =  sqrt(assemble(inner(DGnewerror1,DGnewerror1)*dx))
               NormDic["L2errnewDG1"].append(L2errnewDG1) ;
               NormDic["VherrnewDG1"].append(VherrnewDG1) ;

            if self.params['MS2']:
               u_newDGvec2 = SolDic['UsolDGnew2']
               DGnewerror2 = Function(Vd)
               
               DGnewerror2.vector()[:] = uexaDG.vector()[:] - u_newDGvec2.vector()[:]
               
               VherrnewDG2   =  sqrt(assemble(dg.gh(DGnewerror2,DGnewerror2,grad(DGnewerror2),grad(DGnewerror2),Constant(1)))) #+ bhrhs2
               L2errnewDG2   =  sqrt(assemble(inner(DGnewerror2,DGnewerror2)*dx))
               NormDic["L2errnewDG2"].append(L2errnewDG2) ;
               NormDic["VherrnewDG2"].append(VherrnewDG2) ;
               

        if self.params['REF_TYPE'] == 0:
           NormDic["Dofsvect"].append((mesh.hmax()+mesh.hmin())/2)
        else:
           NormDic["Dofsvect"].append(self.ArgsDic['DOFset'][2]**(-1/mesh.topology().dim()))

        return NormDic