#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 23:01:57 2025

@author: Juan Giraldo
"""

import os
import numpy as np
from dolfin import Mesh, FunctionSpace, Function, Constant

class FileManager:
    def __init__(self, params):
        self.p = params

    def setup_folders(self):
        """
        Create and store folder names in params, ensuring directories exist.
        Returns tuple of key folder paths.
        """
        p = self.p
        # base solution folder
        folder = p['pmain']
        if p['Unsteady']:
            pass  # keep pmain
        if p['Nonlinear']:
            folder += '-NL'
        else:
            folder += '-L'
        p['foldername_sol'] = folder

        # other folders
        p['foldername_con']           = os.path.join(folder, 'Convergence')
        p['foldername_info']          = os.path.join(p['pmain'], 'Info')
        p['foldername_con_time']      = os.path.join(p['pmain'], 'conver_time')
        p['foldername_con_iterpertime']= os.path.join(p['pmain'], 'conver_time', 'iterpertime')
        p['foldername_xml']           = os.path.join(p['pmain'], 'xmlfiles')
        p['foldername_pot']           = os.path.join(p['pmain'], 'potential')

        # make convergence folder
        os.makedirs(p['foldername_con'], exist_ok=True)
        return (
            p['foldername_sol'],
            p['foldername_con'],
            p['foldername_xml'],
            p['foldername_pot'],
            p['foldername_con_time'],
            p['foldername_con_iterpertime']
        )

    def load_solutions(self):
        """
        Load initial (and possibly next-step) CG and DG solutions from XML.
        Returns u_n, u_np, u_np2, uDG_np, uDG_np2
        """
        p = self.p
        base = p['foldername_solution']
        ts0 = p['ts_ini']
        # helper to build path and check
        def check(path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"XML file not found: '{path}'")
            return path

        mesh0 = Mesh(check(f"{base}/timexml/mesh-ts{ts0}.xml"))
        us_n  = FunctionSpace(mesh0, p['TestType'], p['Ptrial'])
        u_n   = Function(us_n, check(f"{base}/timexml/sol-ts{ts0}.xml"))
        u_np  = Function(us_n, check(f"{base}/timexml/sol-ts{ts0}.xml"))
        u_np2 = Constant(0)

        if p['TIM-PM'] == 'BDF2':
            ts1 = ts0 + p['ts_incre']
            mesh1 = Mesh(check(f"{base}/timexml/mesh-ts{ts1}.xml"))
            us_n2 = FunctionSpace(mesh1, p['TestType'], p['Ptrial'])
            u_np2 = Function(us_n2, check(f"{base}/timexml/sol-ts{ts1}.xml"))

        uDG_np  = Constant(0)
        uDG_np2 = Constant(0)
        if p.get('DG'):
            solDG0 = check(f"{base}/timexml/solDG-ts{ts0}.xml")
            uDGs_n = FunctionSpace(mesh0, p['TestType'], p['Ptrial'])
            uDG_np = Function(uDGs_n, solDG0)
            if p['TIM-PM'] == 'BDF2':
                ts1 = ts0 + p['ts_incre']
                solDG1 = check(f"{base}/timexml/solDG-ts{ts1}.xml")
                mesh1 = Mesh(check(f"{base}/timexml/mesh-ts{ts1}.xml"))
                uDGs_n2 = FunctionSpace(mesh1, p['TestType'], p['Ptrial'])
                uDG_np2 = Function(uDGs_n2, solDG1)

        return u_n, u_np, u_np2, uDG_np, uDG_np2



