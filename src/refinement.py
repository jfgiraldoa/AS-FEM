#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 22:15:51 2025

@author: Juan Giraldo
"""

import numpy as np
from dolfin import (
    MeshFunction, refine, FunctionSpace, TestFunction, Function,
    FacetNormal, CellDiameter, grad, assemble, sqrt
)
from dg_forms import DGForms

class RefinementStrategy:

    def __init__(self, argsDic, params):
        self.argsDic = argsDic
        self.params = params

    def refine_mesh(self, mesh, E, g):
        """
        Refine the mesh according to the selected strategy:
          REF_TYPE=0: uniform
          REF_TYPE=1: Dörfler marking
          REF_TYPE=2: fixed fraction marking
        Returns the refined mesh.
        """
        p = self.params
        # boolean marker per cell
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())

        if p['REF_TYPE'] == 1:
            # Dörfler marking
            rgind = np.argsort(g)[::-1]
            threshold = p['REFINE_RATIO']**2 * E**2
            Ntot = mesh.num_cells()
            scum = g[rgind[0]]
            cell_markers[rgind[0]] = True
            cutoff = g[rgind[0]]
            for idx in rgind[1:]:
                scum += g[idx]
                if scum < threshold:
                    cell_markers[idx] = True
                    cutoff = g[idx]
                else:
                    if cutoff - (1 + p['tolref']) * g[idx] < 0:
                        cell_markers[idx] = True
                    else:
                        break
            mesh_refined = refine(mesh, cell_markers)

        elif p['REF_TYPE'] == 0:
            # uniform refinement
            mesh_refined = refine(mesh)

        elif p['REF_TYPE'] == 2:
            # fixed fraction marking
            eta2 = np.array(g)
            eta2_max = eta2.max()
            sum_eta2 = eta2.sum()
            frac = 0.95
            delfrac = 0.05
            marked = np.zeros_like(eta2, dtype=bool)
            sum_marked = 0.0
            while sum_marked < p['REFINE_RATIO'] * sum_eta2:
                new_marked = (~marked) & (eta2 > frac * eta2_max)
                sum_marked += eta2[new_marked].sum()
                marked |= new_marked
                frac -= delfrac
            cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
            cell_markers.array()[:] = marked
            mesh_refined = refine(mesh, cell_markers)

        else:
            raise Exception('No refine type number selected')

        return mesh_refined

    def compute_error(self, Eudis):
        """
        Compute local error indicators and global error norm.

        :param Eudis: DG Function of local indicator values
        :return: E (global error), g (array of local indicators), g_plot (DG0 Function)
        """
        A = self.argsDic
        # build DGForms with mesh normals/diameters
        V = Eudis.function_space()
        mesh = V.mesh()
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)
        dg = DGForms(n, h, A)

        PC2 = FunctionSpace(mesh, "DG", 0)
        c = TestFunction(PC2)
        g_plot = Function(PC2)

        # assemble local indicators
        g_vec = assemble(
            dg.gh(Eudis, Eudis,
                  grad(Eudis), grad(Eudis), c)
        )
        g = np.abs(g_vec)
        g_plot.vector()[:] = g

        # global energy norm
        Ee = g.sum()
        E = sqrt(Ee)
        print(f"Error E = {E:.4g}")
        return E, g, g_plot