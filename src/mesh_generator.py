#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:37:46 2025

@author: Juan Giraldo
"""
from dolfin import IntervalMesh, Measure, RectangleMesh, BoxMesh, Point, Mesh, near, MeshFunction, SubDomain
import pygmsh
import meshio
from pathlib import Path

class MeshGenerator:
    def __init__(self, params):
        self.p = params
    def Create_mesh(self):
        if self.p['mesh_type'] == 'structured':
            if self.p['dim']==1:
               mesh = IntervalMesh(self.p['Nx'], self.p['Linfx'], self.p['Lsupx'])
            elif self.p['dim']==2:
               mesh = RectangleMesh(Point(self.p['Linfx'],self.p['Linfy']), Point(self.p['Lsupx'], self.p['Lsupy']), self.p['Nx'], self.p['Ny'], "crossed")
            elif self.p['dim']==3:
               mesh =  BoxMesh(Point(self.p['Linfx'],self.p['Linfy'],self.p['Linfz']), Point(self.p['Lsupx'],self.p['Lsupy'],self.p['Lsupz']),self.p['Nx'], self.p['Ny'], self.p['Nz'])
            
        elif self.p['mesh_type'] == 'L-shape-3D':
            characteristic_length = 0.2
            with pygmsh.occ.Geometry() as geom:
                big_cube = geom.add_box([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], mesh_size=characteristic_length)
                small_cube = geom.add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], mesh_size=characteristic_length)
                lshape = geom.boolean_difference([big_cube], [small_cube])
                meshio_mesh = geom.generate_mesh(dim=3)

            out = Path(self.p['pmain']+'-L') / "lshape.xml"
            out.parent.mkdir(parents=True, exist_ok=True)
            meshio.write(str(out), meshio_mesh, file_format="dolfin-xml")
            mesh = Mesh(str(out))
        
        else:
                raise Exception('TODO other dim')
        return mesh
    
class BoundaryCreator:
    def __init__(self, p, mesh):
        self.mesh = mesh
        self.p = p

    def create_boundaries(self):
        """
        Return a MeshFunction marking boundary facets:
          0 = default
          1 = left
          2 = top
          3 = right
          4 = bottom
        If p['Constant_tag_BC'] is True, all facets are marked 1.
        """
        mesh, p = self.mesh, self.p

        # Define SubDomains for each side
        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], p['Linfx']) and on_boundary
        class Right(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], p['Lsupx']) and on_boundary
        class Bottom(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], p['Linfy']) and on_boundary
        class Top(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], p['Lsupy']) and on_boundary

        # Initialize facet markers
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)

        if p.get('Constant_tag_BC', False):
            boundaries.set_all(1)
        else:
            # Mark each side with unique tags
            Left().mark(boundaries,   1)
            Top().mark(boundaries,    2)
            Right().mark(boundaries,  3)
            Bottom().mark(boundaries, 4)

        return boundaries
    
class BoundaryMeasures:
    """
    Encapsulates creation of boundary integration measures (ds, dS)
    and sub-measures for Dirichlet, Neumann, Neumann2, Robin conditions.

    :param mesh:        DOLFIN Mesh
    :param boundaries:  MeshFunction marking boundary facets
    :param p:           dict with key 'q_deg' (quadrature degree)
    :param gD_tags:     list of Dirichlet boundary tags
    :param gN_tags:     list of Neumann boundary tags
    :param g2N_tags:    list of second Neumann boundary tags
    :param gR_tags:     list of Robin boundary tags
    """
    def __init__(self, mesh, boundaries, p,
                 gD_tags=None, gN_tags=None, g2N_tags=None, gR_tags=None):
        self.mesh       = mesh
        self.boundaries = boundaries
        self.p          = p
        self.gD_tags    = gD_tags or []
        self.gN_tags    = gN_tags or []
        self.g2N_tags   = g2N_tags or []
        self.gR_tags    = gR_tags or []

        self._build_base_measures()
        self._build_sub_measures()

    def _build_base_measures(self):
        """
        Create the base facet measures ds and dS, with optional quadrature.
        """
        if self.p.get('q_deg'):
            md = {"quadrature_scheme": "default",
                  "quadrature_degree": self.p['q_deg']}
            self.ds  = Measure('ds', domain=self.mesh,
                               metadata=md,
                               subdomain_data=self.boundaries)
            self.dS  = Measure('dS', domain=self.mesh,
                               metadata=md,
                               subdomain_data=self.boundaries)
        else:
            self.ds  = Measure('ds', domain=self.mesh,
                               subdomain_data=self.boundaries)
            self.dS  = Measure('dS', domain=self.mesh,
                               subdomain_data=self.boundaries)

    def _combine_tags(self, base_measure, tags):
        """
        Combine a list of tags into a single sub-measure via addition.
        Returns None if no tags provided.
        """
        if not tags:
            return None
        m = base_measure(tags[0])
        for tag in tags[1:]:
            m += base_measure(tag)
        return m

    def _build_sub_measures(self):
        """
        Build sub-measures for each boundary condition type.
        """
        self.dsD  = self._combine_tags(self.ds,  self.gD_tags)
        self.dsN  = self._combine_tags(self.ds,  self.gN_tags)
        self.dsN2 = self._combine_tags(self.ds,  self.g2N_tags)
        self.dsR  = self._combine_tags(self.ds,  self.gR_tags)

    def get_measures(self):
        return self.ds, self.dS,self.dsD, self.dsN,self.dsN2, self.dsR