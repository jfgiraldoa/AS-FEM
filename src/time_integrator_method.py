#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 22:48:51 2025

@author: Juan Giraldo
"""
class TimeIntegratorMethod:
    """
    time-integration coefficients
    :param method: str, one of 'BDF2', 'BE'
    :param params: dict containing 'dt-PM'
    """
    def __init__(self, method, params):
        self.method = method
        self.params = params

    def coefficients(self):
        """
        Return the tuple (C1, C2, Crhs1, Crhs2, Crhs3)
        corresponding to the chosen time integrator.
        """
        dt = self.params.get('dt-PM')
        if self.method == 'BDF2':
            C1 = 1.0
            C2 = 2.0/3.0 * dt
            Crhs1 = 4.0/3.0
            Crhs2 = -1.0/3.0
            Crhs3 = 2.0/3.0 * dt
        elif self.method == 'BE':
            C1 = 1.0
            C2 = dt
            Crhs1 = 0.0
            Crhs2 = 1.0
            Crhs3 = dt
        else:
            raise ValueError(f"Unsupported time integrator: {self.method}")
        return C1, C2, Crhs1, Crhs2, Crhs3

