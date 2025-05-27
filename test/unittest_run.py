# 2025 Juan Felipe Giraldo. Licensed under the MIT License.

import unittest
import sys
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from contextlib import contextmanager
from test_util import prepare_test_environment
prepare_test_environment()

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

from unittest_cases import (
    test_poisson_FEM_solution,
    test_steady_Fichera_vms,
    test_steady_Heterogeneous_vms,
    test_steady_Anisotropic_vms,
    test_steady_ErikssonJhonsson,
    test_unsteady_ErikssonJhonssonBE,
    test_unsteady_ErikssonJhonssonBDF2,
    test_steady_Burgers_vms
)

class ProgressTestResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.testsRunProgress = getattr(self, 'testsRunProgress', 0) + 1
        total = self.testsRun + self.testsLeft()
        print(f"{self.testsRunProgress}/{total} - {test._testMethodName}")

    def testsLeft(self):
        return self.test_count - self.testsRunProgress

class ProgressTestRunner(unittest.TextTestRunner):
    def run(self, test):
        result = self._makeResult()
        result.test_count = test.countTestCases()
        return super().run(test)

class TestASFEMCases(unittest.TestCase):

    def test_poisson(self):
        with suppress_output():
            result = test_poisson_FEM_solution()
        self.assertAlmostEqual(result, 0.2471405369044419, places=10)

    def test_fichera(self):
        with suppress_output():
            result = test_steady_Fichera_vms()
        self.assertAlmostEqual(result, 0.0036270308223143605, places=12)

    def test_heterogeneous(self):
        with suppress_output():
            result = test_steady_Heterogeneous_vms()
        self.assertAlmostEqual(result, 0.0025428740360954874, places=12)

    def test_anisotropic(self):
        with suppress_output():
            result = test_steady_Anisotropic_vms()
        self.assertAlmostEqual(result, 0.0004639546948881584, places=12)

    def test_eriksson(self):
        with suppress_output():
            result = test_steady_ErikssonJhonsson()
        self.assertAlmostEqual(result, 0.016434803708377115, places=12)

    def test_unsteady_be(self):
        with suppress_output():
            result = test_unsteady_ErikssonJhonssonBE()
        self.assertAlmostEqual(result, 0.09013095732710719, places=12)

    def test_unsteady_bdf2(self):
        with suppress_output():
            result = test_unsteady_ErikssonJhonssonBDF2()
        self.assertAlmostEqual(result, 0.059916400178857125, places=7)

    def test_burgers(self):
        with suppress_output():
            result = test_steady_Burgers_vms()
        self.assertAlmostEqual(result, 0.08007984881050467, places=12)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestASFEMCases)
    runner = ProgressTestRunner(verbosity=2)
    runner.run(suite)
    #sys.exit(0) #comment when using spyder
