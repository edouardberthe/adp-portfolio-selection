import unittest
from abc import ABCMeta, abstractmethod
from random import randint

from data import A, MeanReturns
from entities.portfolio import Portfolio
from generator import generate_gaussian, generate_t
from markowitz import Markowitz
from scenarios_based.models import CVaR, GMD, MAD, Minimax, SemiMAD, VaR


class TestCaseWrapper(object):
    """
    All the base plot_test cases are defined inside this class, because we do not want them to be read as tests
    themselves (because they should be overriden).
    """

    class ModelBaseTestCase(unittest.TestCase):
        """
        Base plot_test case for all the portfolio models (Markowitz + Scenarios-based).
        """

        __metaclass__ = ABCMeta

        @abstractmethod
        def Model(self):
            """Returns the class of the Model tested (should be overridden)."""
            return None

        def createModel(self, *args, **kwargs):
            """
            Will be overriden by the scenarios-based portfolio models (which requires scenarios / probas).
            :rtype: entities.PortfolioOptimizer
            """
            return self.Model(*args, **kwargs)

        def test_init(self):
            self.assertIsInstance(self.createModel(), self.Model)

        def test_get_portfolios(self):
            self.assertIsInstance(self.createModel().optimize().getPortfolio(), Portfolio)

        def test_Wmin(self):
            Wmin = 0.1
            m = self.createModel(Wmin=Wmin)
            self.assertIsInstance(
                m.optimize().objval,
                float
            )
            for a in A:
                if m._X[a].x > 0.5:
                    self.assertGreaterEqual(m._W[a].x, Wmin - 1e-7)

        def test_Wmax(self):
            Wmax = 0.1
            m = self.createModel(Wmax=Wmax)
            self.assertIsInstance(m.optimize().objval, float)
            for a in A:
                self.assertLessEqual(m._W[a].x, Wmax)

        def test_Nmax(self):
            Nmax = 3
            m = self.createModel(Nmax=Nmax)
            self.assertIsInstance(m.optimize().objval, float)
            nbAssets = 0
            for a in A:
                if m._W[a].x > 1e-7:
                    nbAssets += 1
            self.assertLessEqual(nbAssets, Nmax)

    class ScenariosBasedModelBaseTestCase(ModelBaseTestCase):

        __metaclass__ = ABCMeta

        def setUp(self):
            self.seed = 5
            self.scenarios, self.probas = generate_gaussian(100, seed=self.seed)

        def createModel(self, *args, **kwargs):
            """
            :rtype: linear.models.model.ScenariosBasedPortfolioModel
            """
            return self.Model(self.scenarios, self.probas, *args, **kwargs)

        def test_reconfigure(self):
            """
            Checks that reconfiguring the model with new scenarios/probas gives the same output as creating a new Model.
            """
            s, p = self.scenarios, self.probas
            s2, p2 = generate_gaussian(len(p), seed=self.seed + 1 if self.seed is not None else None)

            self.assertEqual(
                self.Model(s, p).update().reconfigure(s2, p2).optimize().objval,
                self.Model(s2, p2).optimize().objval
            )

    class GeneratorBaseTestCase(unittest.TestCase):
        """
        This class should be overriden by all the plot_test cases of generators. Tests the scenarios generators over the
        basics functions.
        """

        __metaclass__ = ABCMeta

        @abstractmethod
        def generator(self):
            """Should be overriden. Returns the scenarios generator."""
            raise NotImplementedError

        def test_shape(self):
            N = randint(1000, 100000)
            scenarios, probas = self.generator(N)
            self.assertEqual(scenarios.shape, (N, len(A)))
            self.assertEqual(probas.shape, (N,))

        def test_mean(self):
            N = 1000000
            s, p = self.generator(N)
            self.assertAlmostEqual((s.mean(axis=0) * 252 - MeanReturns).mean(), 0, delta=0.1)


class TestMAD(TestCaseWrapper.ScenariosBasedModelBaseTestCase):

    @property
    def Model(self):
        return MAD


class TestSemiMAD(TestCaseWrapper.ScenariosBasedModelBaseTestCase):

    @property
    def Model(self):
        return SemiMAD


class TestGMD(TestCaseWrapper.ScenariosBasedModelBaseTestCase):

    @property
    def Model(self):
        return GMD


class TestMinimax(TestCaseWrapper.ScenariosBasedModelBaseTestCase):

    @property
    def Model(self):
        return Minimax


class TestVaR(TestCaseWrapper.ScenariosBasedModelBaseTestCase):

    @property
    def Model(self):
        return VaR


class TestCVaR(TestCaseWrapper.ScenariosBasedModelBaseTestCase):

    @property
    def Model(self):
        return CVaR


class TestMarkowitz(TestCaseWrapper.ModelBaseTestCase):

    @property
    def Model(self):
        return Markowitz


class TestMADAndSemiMAD(unittest.TestCase):

    def test_same_output(self):
        """Checks that the MAD and the SemiMAD give the same output portfolio and half of the objective value."""
        s, p = generate_gaussian()
        m = MAD(s, p).optimize()
        m2 = SemiMAD(s, p).optimize()

        self.assertAlmostEqual((m.getPortfolio() - m2.getPortfolio()).max(), 0, places=4)
        self.assertAlmostEqual(m.objval, 2 * m2.objval)


class GaussianGeneratorTestCase(TestCaseWrapper.GeneratorBaseTestCase):

    @property
    def generator(self):
        return generate_gaussian


class StudentGeneratorTestCase(TestCaseWrapper.GeneratorBaseTestCase):

    @property
    def generator(self):
        return generate_t


class Test(unittest.TestCase):

    def test_coucou(self):
        self.assertEqual(3, 3)

if __name__ == '__main__':
    unittest.main()
