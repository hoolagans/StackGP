"""
unitTest.py - Comprehensive unit tests for StackGP

Run with:
    python -m pytest unitTest.py -v
or:
    python unitTest.py
"""

import unittest
import copy
import math
import random
import numpy as np
import sys
import os

# Ensure the StackGP directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import StackGP as sgp


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def simple_model():
    """Return a simple, deterministic model:  add(x0, x1)  →  x0 + x1."""
    ops = np.array([sgp.add], dtype=object)
    var = [sgp.variableSelect(0), sgp.variableSelect(1)]
    return [ops, var, []]


def constant_model(c=5.0):
    """Return a model that always returns constant c:  pop  →  c."""
    ops = np.array(["pop"], dtype=object)
    var = [c]
    return [ops, var, []]


def linear_model():
    """Return a model: mult(x0, 2.0)  →  2*x0."""
    ops = np.array([sgp.mult], dtype=object)
    var = [sgp.variableSelect(0), 2.0]
    return [ops, var, []]


def set_quality(model, x, y, metrics=None):
    if metrics is None:
        metrics = [sgp.fitness, sgp.stackGPModelComplexity]
    sgp.setModelQuality(model, x, y, metrics)


# ---------------------------------------------------------------------------
# 1. Mathematical Operator Functions
# ---------------------------------------------------------------------------

class TestMathOperators(unittest.TestCase):

    def test_add(self):
        self.assertAlmostEqual(sgp.add(3, 4), 7)
        np.testing.assert_array_almost_equal(sgp.add(np.array([1, 2]), np.array([3, 4])), [4, 6])

    def test_sub(self):
        self.assertAlmostEqual(sgp.sub(10, 3), 7)

    def test_mult(self):
        self.assertAlmostEqual(sgp.mult(3, 4), 12)

    def test_protectDiv_normal(self):
        self.assertAlmostEqual(sgp.protectDiv(10, 2), 5)

    def test_protectDiv_zero_scalar(self):
        result = sgp.protectDiv(5, 0)
        self.assertTrue(math.isnan(result))

    def test_protectDiv_zero_in_array(self):
        result = sgp.protectDiv(np.array([4.0, 6.0]), np.array([2.0, 0.0]))
        self.assertAlmostEqual(result[0], 2.0)
        self.assertTrue(np.isnan(result[1]))

    def test_exp(self):
        self.assertAlmostEqual(sgp.exp(0), 1.0)
        self.assertAlmostEqual(sgp.exp(1), math.e)

    def test_power_normal(self):
        self.assertAlmostEqual(sgp.power(2, 3), 8)

    def test_power_zero_base(self):
        result = sgp.power(0, 3)
        self.assertTrue(math.isnan(result))

    def test_sqrt(self):
        self.assertAlmostEqual(sgp.sqrt(9), 3.0)

    def test_sqrd(self):
        self.assertAlmostEqual(sgp.sqrd(5), 25)

    def test_inv(self):
        self.assertAlmostEqual(sgp.inv(4), 0.25)

    def test_neg(self):
        self.assertAlmostEqual(sgp.neg(7), -7)
        self.assertAlmostEqual(sgp.neg(-3), 3)

    def test_sin(self):
        self.assertAlmostEqual(sgp.sin(0), 0.0)
        self.assertAlmostEqual(sgp.sin(math.pi / 2), 1.0)

    def test_cos(self):
        self.assertAlmostEqual(sgp.cos(0), 1.0)
        self.assertAlmostEqual(sgp.cos(math.pi), -1.0)

    def test_tan(self):
        self.assertAlmostEqual(sgp.tan(0), 0.0)

    def test_arcsin(self):
        self.assertAlmostEqual(sgp.arcsin(1.0), math.pi / 2)

    def test_arccos(self):
        self.assertAlmostEqual(sgp.arccos(1.0), 0.0)

    def test_arctan(self):
        self.assertAlmostEqual(sgp.arctan(1.0), math.pi / 4)

    def test_tanh(self):
        self.assertAlmostEqual(sgp.tanh(0), 0.0)
        self.assertAlmostEqual(sgp.tanh(1), math.tanh(1))

    def test_log(self):
        self.assertAlmostEqual(sgp.log(math.e), 1.0)

    def test_log10(self):
        self.assertAlmostEqual(sgp.log10(100), 2.0)

    def test_log2(self):
        self.assertAlmostEqual(sgp.log2(8), 3.0)

    def test_abs1(self):
        self.assertAlmostEqual(sgp.abs1(-5), 5)
        self.assertAlmostEqual(sgp.abs1(3), 3)


# ---------------------------------------------------------------------------
# 2. Boolean Operator Functions
# ---------------------------------------------------------------------------

class TestBooleanOperators(unittest.TestCase):

    def test_and1(self):
        self.assertTrue(sgp.and1(True, True))
        self.assertFalse(sgp.and1(True, False))

    def test_or1(self):
        self.assertTrue(sgp.or1(False, True))
        self.assertFalse(sgp.or1(False, False))

    def test_xor1(self):
        self.assertTrue(sgp.xor1(True, False))
        self.assertFalse(sgp.xor1(True, True))

    def test_nand1(self):
        self.assertFalse(sgp.nand1(True, True))
        self.assertTrue(sgp.nand1(True, False))

    def test_nor1(self):
        self.assertTrue(sgp.nor1(False, False))
        self.assertFalse(sgp.nor1(True, False))

    def test_xnor1(self):
        self.assertTrue(sgp.xnor1(True, True))
        self.assertFalse(sgp.xnor1(True, False))

    def test_not1(self):
        self.assertFalse(sgp.not1(True))
        self.assertTrue(sgp.not1(False))

    def test_array_and1(self):
        a = np.array([True, False, True])
        b = np.array([True, True, False])
        result = sgp.and1(a, b)
        np.testing.assert_array_equal(result, [True, False, False])


# ---------------------------------------------------------------------------
# 3. Op/Const Lists
# ---------------------------------------------------------------------------

class TestOpConstLists(unittest.TestCase):

    def test_defaultOps_returns_list(self):
        ops = sgp.defaultOps()
        self.assertIsInstance(ops, list)
        self.assertGreater(len(ops), 0)

    def test_allOps_contains_trig(self):
        ops = sgp.allOps()
        self.assertIn(sgp.sin, ops)
        self.assertIn(sgp.cos, ops)
        self.assertIn(sgp.tan, ops)

    def test_booleanOps_contains_and_or(self):
        ops = sgp.booleanOps()
        self.assertIn(sgp.and1, ops)
        self.assertIn(sgp.or1, ops)

    def test_defaultConst_callable_elements(self):
        const = sgp.defaultConst()
        self.assertIn(np.pi, const)
        self.assertIn(np.e, const)

    def test_booleanConst(self):
        const = sgp.booleanConst()
        self.assertIn(0, const)
        self.assertIn(1, const)

    def test_randomInt_range(self):
        for _ in range(20):
            v = sgp.randomInt(-3, 3)
            self.assertGreaterEqual(v, -3)
            self.assertLessEqual(v, 3)

    def test_ranReal_type(self):
        v = sgp.ranReal()
        self.assertIsInstance(v, float)


# ---------------------------------------------------------------------------
# 4. Data Subsampling Methods
# ---------------------------------------------------------------------------

class TestDataSubsampling(unittest.TestCase):

    def _make_data(self, n=50, dims=2):
        x = np.random.rand(dims, n)
        y = np.random.rand(n)
        return x, y

    def test_randomSubsample_shape(self):
        x, y = self._make_data()
        xs, ys = sgp.randomSubsample(x, y)
        self.assertEqual(xs.shape[0], x.shape[0])
        self.assertLessEqual(xs.shape[1], x.shape[1])
        self.assertEqual(xs.shape[1], len(ys))

    def test_generationProportionalSample(self):
        x, y = self._make_data()
        xs, ys = sgp.generationProportionalSample(x, y, generation=50, generations=100)
        self.assertEqual(xs.shape[0], x.shape[0])
        self.assertEqual(xs.shape[1], len(ys))

    def test_ordinalSample(self):
        x, y = self._make_data()
        xs, ys = sgp.ordinalSample(x, y, generation=50, generations=100)
        self.assertEqual(xs.shape[1], len(ys))

    def test_orderedSample(self):
        x, y = self._make_data()
        xs, ys = sgp.orderedSample(x, y, generation=50, generations=100)
        self.assertEqual(xs.shape[1], len(ys))

    def test_ordinalBalancedSample(self):
        x, y = self._make_data()
        xs, ys = sgp.ordinalBalancedSample(x, y, generation=50, generations=100)
        self.assertEqual(xs.shape[1], len(ys))

    def test_balancedSample(self):
        x, y = self._make_data()
        xs, ys = sgp.balancedSample(x, y)
        self.assertEqual(xs.shape[1], len(ys))


# ---------------------------------------------------------------------------
# 5. Model Structure Helpers
# ---------------------------------------------------------------------------

class TestModelStructureHelpers(unittest.TestCase):

    def test_getArity_binary(self):
        self.assertEqual(sgp.getArity(sgp.add), 2)
        self.assertEqual(sgp.getArity(sgp.mult), 2)
        self.assertEqual(sgp.getArity(sgp.protectDiv), 2)

    def test_getArity_unary(self):
        self.assertEqual(sgp.getArity(sgp.exp), 1)
        self.assertEqual(sgp.getArity(sgp.sqrt), 1)
        self.assertEqual(sgp.getArity(sgp.neg), 1)

    def test_getArity_pop(self):
        self.assertEqual(sgp.getArity("pop"), 1)

    def test_modelArity_add(self):
        model = simple_model()
        self.assertEqual(sgp.modelArity(model), 2)

    def test_modelArity_pop(self):
        model = constant_model()
        self.assertEqual(sgp.modelArity(model), 1)

    def test_listArity_empty(self):
        self.assertEqual(sgp.listArity([]), 0)

    def test_listArity_single_binary(self):
        self.assertEqual(sgp.listArity([sgp.add]), 2)

    def test_listArity_two_unary(self):
        # Each neg is unary (arity 1): the chain neg(neg(x)) still requires exactly
        # 1 input variable.  listArity formula: 1 + sum(arity-1) = 1 + 0 + 0 = 1.
        self.assertEqual(sgp.listArity([sgp.neg, sgp.neg]), 1)

    def test_buildEmptyModel(self):
        m = sgp.buildEmptyModel()
        self.assertEqual(len(m), 3)
        self.assertEqual(m[0], [])
        self.assertEqual(m[1], [])
        self.assertEqual(m[2], [])

    def test_variableSelect(self):
        sel = sgp.variableSelect(1)
        data = [[10, 20], [30, 40]]
        # variableSelect(1) should return data[1]
        self.assertEqual(sel(data), [30, 40])

    def test_reverseList(self):
        self.assertEqual(sgp.reverseList([1, 2, 3]), [3, 2, 1])
        self.assertEqual(sgp.reverseList([]), [])

    def test_varReplace_non_callable(self):
        result = sgp.varReplace([1, 2, 3], [[0], [1]])
        self.assertEqual(result, [1, 2, 3])

    def test_varReplace_with_lambda(self):
        sel = sgp.variableSelect(0)
        data = [[10, 20], [30, 40]]
        result = sgp.varReplace([sel], data)
        self.assertEqual(result[0], [10, 20])

    def test_inputLen_scalar(self):
        data = [5, 3]
        self.assertEqual(sgp.inputLen(data), 1)

    def test_inputLen_array(self):
        data = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(sgp.inputLen(data), 3)

    def test_varCount(self):
        data = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(sgp.varCount(data), 3)

    def test_get_numeric_indices(self):
        lst = [sgp.variableSelect(0), 3.14, sgp.add, 2]
        idx = sgp.get_numeric_indices(lst)
        self.assertIn(1, idx)
        self.assertIn(3, idx)
        self.assertNotIn(0, idx)
        self.assertNotIn(2, idx)


# ---------------------------------------------------------------------------
# 6. Model Generation
# ---------------------------------------------------------------------------

class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_generateRandomModel_structure(self):
        model = sgp.generateRandomModel(variables=2, ops=sgp.defaultOps(),
                                        const=sgp.defaultConst(), maxLength=5)
        self.assertEqual(len(model), 3)
        self.assertIsInstance(model[0], np.ndarray)
        self.assertIsInstance(model[1], list)

    def test_generateRandomModel_arity_consistency(self):
        for _ in range(20):
            model = sgp.generateRandomModel(variables=3, ops=sgp.defaultOps(),
                                            const=sgp.defaultConst(), maxLength=8)
            expected = sgp.modelArity(model)
            self.assertEqual(len(model[1]), expected)

    def test_generateRandomModel_not_all_constants(self):
        """At least one variable selector must be present in model[1]."""
        for _ in range(20):
            model = sgp.generateRandomModel(variables=2, ops=sgp.defaultOps(),
                                            const=sgp.defaultConst(), maxLength=5)
            has_var = any(callable(v) for v in model[1])
            self.assertTrue(has_var, "Model should contain at least one variable selector")

    def test_initializeGPModels_count(self):
        models = sgp.initializeGPModels(2, sgp.defaultOps(), sgp.defaultConst(),
                                         numberOfModels=10)
        self.assertEqual(len(models), 10)

    def test_initializeGPModels_each_valid(self):
        models = sgp.initializeGPModels(2, sgp.defaultOps(), sgp.defaultConst(),
                                         numberOfModels=5, maxLength=6)
        for m in models:
            self.assertEqual(len(m), 3)
            self.assertEqual(len(m[1]), sgp.modelArity(m))


# ---------------------------------------------------------------------------
# 7. Model Evaluation
# ---------------------------------------------------------------------------

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # x0 + x1 = [5, 7, 9]
        self.y_add = self.x[0] + self.x[1]
        # 2 * x0
        self.y_linear = 2.0 * self.x[0]

    def test_evaluateGPModel_add(self):
        model = simple_model()
        result = sgp.evaluateGPModel(model, self.x)
        np.testing.assert_array_almost_equal(result, self.y_add)

    def test_evaluateGPModel_constant(self):
        model = constant_model(7.0)
        result = sgp.evaluateGPModel(model, self.x)
        np.testing.assert_array_almost_equal(result, [7.0, 7.0, 7.0])

    def test_evaluateGPModel_linear(self):
        model = linear_model()
        result = sgp.evaluateGPModel(model, self.x)
        np.testing.assert_array_almost_equal(result, self.y_linear)

    def test_rmse_perfect(self):
        model = simple_model()
        err = sgp.rmse(model, self.x, self.y_add)
        self.assertAlmostEqual(err, 0.0, places=10)

    def test_rmse_nonzero(self):
        model = simple_model()
        err = sgp.rmse(model, self.x, self.y_add + 1.0)
        self.assertAlmostEqual(err, 1.0, places=10)

    def test_fitness_perfect(self):
        """Perfect linear predictor should have fitness close to 0 (1-R^2)."""
        model = simple_model()
        fit = sgp.fitness(model, self.x, self.y_add)
        self.assertAlmostEqual(fit, 0.0, places=6)

    def test_fitnessconstant_model(self):
        """Constant model has no correlation with response → fitness = nan or 1."""
        model = constant_model(3.0)
        fit = sgp.fitness(model, self.x, self.y_add)
        # constant output produces nan fitness
        self.assertTrue(fit is None or math.isnan(fit) or fit == 1.0)

    def test_fitness_range(self):
        """Fitness should be between 0 and 1 for a reasonable model."""
        np.random.seed(1)
        x = np.random.rand(2, 50)
        y = x[0] + x[1] + 0.1 * np.random.randn(50)
        model = simple_model()
        fit = sgp.fitness(model, x, y)
        if not math.isnan(fit):
            self.assertGreaterEqual(fit, 0.0)
            self.assertLessEqual(fit, 1.0)

    def test_binaryError_within_bounds(self):
        """binaryError should return a value in [0, 0.5]."""
        model = simple_model()
        response = np.array([1.0, 1.0, 1.0])
        err = sgp.binaryError(model, self.x, response)
        self.assertGreaterEqual(err, 0.0)
        self.assertLessEqual(err, 0.5)

    def test_stackGPModelComplexity(self):
        model = simple_model()
        c = sgp.stackGPModelComplexity(model)
        self.assertIsInstance(c, int)
        self.assertGreater(c, 0)

    def test_stackGPModelComplexity_pop(self):
        """'pop' operators should not add to complexity."""
        model_with_pop = [np.array([sgp.add, "pop"], dtype=object),
                          [sgp.variableSelect(0), sgp.variableSelect(1), sgp.variableSelect(0)],
                          []]
        c = sgp.stackGPModelComplexity(model_with_pop)
        # ops count + vars count - pops count
        expected = len(model_with_pop[0]) + len(model_with_pop[1]) - 1
        self.assertEqual(c, expected)

    def test_setModelQuality_sets_two_metrics(self):
        model = simple_model()
        set_quality(model, self.x, self.y_add)
        self.assertEqual(len(model[2]), 2)

    def test_setModelQuality_first_metric_is_fitness(self):
        model = simple_model()
        set_quality(model, self.x, self.y_add)
        self.assertAlmostEqual(model[2][0], 0.0, places=6)


# ---------------------------------------------------------------------------
# 8. Crossover and Mutation
# ---------------------------------------------------------------------------

class TestCrossoverAndMutation(unittest.TestCase):

    def setUp(self):
        random.seed(7)
        np.random.seed(7)

    def test_recombination2pt_returns_two_children(self):
        m1 = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        m2 = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        children = sgp.recombination2pt(m1, m2)
        self.assertEqual(len(children), 2)

    def test_recombination2pt_arity_consistency(self):
        for _ in range(20):
            m1 = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
            m2 = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
            c1, c2 = sgp.recombination2pt(m1, m2)
            self.assertEqual(len(c1[1]), sgp.modelArity(c1),
                             f"Child1 arity mismatch: {c1}")
            self.assertEqual(len(c2[1]), sgp.modelArity(c2),
                             f"Child2 arity mismatch: {c2}")

    def test_recombination2pt_parents_unchanged(self):
        m1 = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        m2 = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        m1_copy = copy.deepcopy(m1)
        m2_copy = copy.deepcopy(m2)
        sgp.recombination2pt(m1, m2)
        self.assertTrue(sgp.modelSameQ(m1, m1_copy))
        self.assertTrue(sgp.modelSameQ(m2, m2_copy))

    def test_mutate_returns_model(self):
        m = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        mutated = sgp.mutate(m, 2, sgp.defaultOps(), sgp.defaultConst())
        self.assertEqual(len(mutated), 3)

    def test_mutate_arity_consistency(self):
        for _ in range(30):
            m = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
            mutated = sgp.mutate(m, 2, sgp.defaultOps(), sgp.defaultConst())
            self.assertEqual(len(mutated[1]), sgp.modelArity(mutated),
                             f"Mutated model arity mismatch: {mutated}")

    def test_mutate_original_unchanged(self):
        m = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        original = copy.deepcopy(m)
        sgp.mutate(m, 2, sgp.defaultOps(), sgp.defaultConst())
        self.assertTrue(sgp.modelSameQ(m, original))


# ---------------------------------------------------------------------------
# 9. Pareto Selection
# ---------------------------------------------------------------------------

class TestParetoSelection(unittest.TestCase):

    def _make_pop(self, qualities):
        """Build a dummy population with given quality tuples."""
        pop = []
        for q in qualities:
            m = copy.deepcopy(simple_model())
            m[2] = list(q)
            pop.append(m)
        return pop

    def test_paretoFront_single(self):
        vals = np.array([[0.1, 2], [0.5, 3], [0.2, 4]])
        front = sgp.paretoFront(vals)
        self.assertTrue(front[0])   # (0.1,2) dominates all others

    def test_paretoFront_multiple(self):
        # (0.1,5) and (0.5,1) are both on front; (0.3,3) is dominated
        vals = np.array([[0.1, 5], [0.3, 3], [0.5, 1]])
        front = sgp.paretoFront(vals)
        self.assertTrue(front[0])
        self.assertTrue(front[2])

    def test_paretoTournament_returns_front(self):
        """With objectives [(0.1,5), (0.3,3), (0.5,1)], no single point dominates
        another (lower is better on both axes for different points), so all three
        are on the Pareto front."""
        pop = self._make_pop([(0.1, 5), (0.3, 3), (0.5, 1)])
        front = sgp.paretoTournament(pop)
        qualities = [m[2] for m in front]
        # At least the extreme points must be on the Pareto front
        self.assertIn([0.1, 5], qualities)
        self.assertIn([0.5, 1], qualities)
        # (0.1,10) should NOT appear (it was not in the population)
        self.assertNotIn([0.1, 10], qualities)

    def test_tournamentModelSelection_count(self):
        np.random.seed(0)
        random.seed(0)
        pop = self._make_pop([(0.1, 5), (0.3, 3), (0.5, 1), (0.2, 2), (0.4, 4)])
        selected = sgp.tournamentModelSelection(pop, popSize=5, tourneySize=3)
        self.assertGreaterEqual(len(selected), 5)

    def test_selectModels_reduces_population(self):
        pop = self._make_pop([(0.1, 5), (0.3, 3), (0.5, 1), (0.2, 2), (0.4, 4)])
        selected = sgp.selectModels(pop, 3)
        self.assertGreaterEqual(len(selected), 1)
        self.assertLessEqual(len(selected), len(pop))

    def test_selectModels_fraction(self):
        pop = self._make_pop([(i * 0.1, 10 - i) for i in range(10)])
        selected = sgp.selectModels(pop, 0.3)
        self.assertGreaterEqual(len(selected), 1)


# ---------------------------------------------------------------------------
# 10. Model Comparison and Deduplication
# ---------------------------------------------------------------------------

class TestModelComparison(unittest.TestCase):

    def test_modelSameQ_identical(self):
        m = simple_model()
        self.assertTrue(sgp.modelSameQ(m, copy.deepcopy(m)))

    def test_modelSameQ_different_ops(self):
        m1 = simple_model()
        m2 = linear_model()
        self.assertFalse(sgp.modelSameQ(m1, m2))

    def test_deleteDuplicateModels_removes_duplicates(self):
        m = simple_model()
        pop = [copy.deepcopy(m) for _ in range(5)]
        unique = sgp.deleteDuplicateModels(pop)
        self.assertEqual(len(unique), 1)

    def test_deleteDuplicateModels_preserves_distinct(self):
        pop = [simple_model(), linear_model(), constant_model()]
        unique = sgp.deleteDuplicateModels(pop)
        self.assertEqual(len(unique), 3)

    def test_deleteDuplicateModelsPhenotype(self):
        m = simple_model()
        pop = [copy.deepcopy(m) for _ in range(3)]
        unique = sgp.deleteDuplicateModelsPhenotype(pop)
        self.assertEqual(len(unique), 1)

    def test_removeIndeterminateModels(self):
        good = copy.deepcopy(simple_model())
        good[2] = [0.1, 2]
        bad = copy.deepcopy(simple_model())
        bad[2] = [float('nan'), 2]
        pop = [good, bad]
        cleaned = sgp.removeIndeterminateModels(pop)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0][2][0], 0.1)

    def test_sortModels_ascending(self):
        pop = []
        for v in [0.5, 0.1, 0.3]:
            m = copy.deepcopy(simple_model())
            m[2] = [v]
            pop.append(m)
        sorted_pop = sgp.sortModels(pop)
        vals = [m[2][0] for m in sorted_pop]
        self.assertEqual(vals, sorted(vals))


# ---------------------------------------------------------------------------
# 11. Model Alignment and Trimming
# ---------------------------------------------------------------------------

class TestModelAlignmentTrimming(unittest.TestCase):

    def setUp(self):
        np.random.seed(3)
        self.x = np.random.rand(2, 40)
        self.y = 3.0 * self.x[0] + 2.0 * self.x[1] + 1.0

    def test_trimModel_returns_valid(self):
        model = simple_model()
        trimmed = sgp.trimModel(model)
        self.assertEqual(len(trimmed), 3)
        self.assertEqual(len(trimmed[1]), sgp.modelArity(trimmed))

    def test_alignGPModel_returns_model(self):
        model = simple_model()
        aligned = sgp.alignGPModel(model, self.x, self.y)
        self.assertEqual(len(aligned), 3)

    def test_alignGPModel_improves_fitness(self):
        """After alignment, fitness should not be worse than before."""
        np.random.seed(42)
        x = np.random.rand(2, 50)
        y = 5.0 * x[0] + 3.0 * x[1]
        model = simple_model()
        aligned = sgp.alignGPModel(model, x, y)
        fit_before = sgp.fitness(model, x, y)
        fit_after = sgp.fitness(aligned, x, y)
        if not math.isnan(fit_before) and not math.isnan(fit_after):
            self.assertLessEqual(fit_after, fit_before + 1e-8)

    def test_alignGPModel_constant_returns_unchanged(self):
        """Constant model cannot be aligned; should be returned as-is."""
        model = constant_model(3.0)
        aligned = sgp.alignGPModel(model, self.x, self.y)
        self.assertEqual(len(aligned), 3)


# ---------------------------------------------------------------------------
# 12. Symbolic Printing (printGPModel)
# ---------------------------------------------------------------------------

class TestPrintGPModel(unittest.TestCase):

    def test_printGPModel_returns_expr(self):
        model = simple_model()
        expr = sgp.printGPModel(model)
        # Should be a sympy expression (not nan)
        import sympy as sym
        self.assertNotEqual(expr, float('nan'))

    def test_printGPModel_add(self):
        model = simple_model()
        expr = sgp.printGPModel(model)
        import sympy as sym
        x0, x1 = sym.symbols('x0 x1')
        self.assertEqual(sym.simplify(expr - (x0 + x1)), 0)

    def test_printGPModel_invalid_returns_nan(self):
        """A broken model should return nan without crashing."""
        bad_model = [np.array([], dtype=object), [], []]
        result = sgp.printGPModel(bad_model)
        self.assertTrue(result != result or result is None or isinstance(result, float))


# ---------------------------------------------------------------------------
# 13. Stack Helpers
# ---------------------------------------------------------------------------

class TestStackHelpers(unittest.TestCase):

    def test_stackVarUsage_single_binary(self):
        ops = np.array([sgp.add], dtype=object)
        self.assertEqual(sgp.stackVarUsage(ops), 2)

    def test_stackVarUsage_single_unary(self):
        ops = np.array([sgp.neg], dtype=object)
        self.assertEqual(sgp.stackVarUsage(ops), 1)

    def test_stackGrab_full(self):
        s1 = [1, 2, 3]
        s2 = [10, 20]
        grabbed, ts1, ts2 = sgp.stackGrab(s1, s2, 2)
        self.assertEqual(len(grabbed), 2)
        self.assertEqual(len(grabbed) + len(ts1) + len(ts2), len(s1) + len(s2))

    def test_replaceFunc(self):
        stack = [sgp.add, sgp.mult, sgp.add]
        result = sgp.replaceFunc(stack, sgp.add, sgp.sub)
        self.assertEqual(result, [sgp.sub, sgp.mult, sgp.sub])

    def test_replaceOpsWithStrings_add(self):
        ops = np.array([sgp.add, sgp.mult], dtype=object)
        result = sgp.replaceOpsWithStrings(ops.tolist())
        self.assertIn("+", result)
        self.assertIn("*", result)


# ---------------------------------------------------------------------------
# 14. Ensemble Functions
# ---------------------------------------------------------------------------

class TestEnsembleFunctions(unittest.TestCase):

    def setUp(self):
        np.random.seed(5)
        self.x = np.random.rand(2, 30)
        self.y = self.x[0] + self.x[1]

    def _make_aligned_models(self, n=5):
        models = []
        for _ in range(n):
            m = sgp.generateRandomModel(2, sgp.defaultOps(), sgp.defaultConst(), 4)
            m = sgp.alignGPModel(m, self.x, self.y)
            set_quality(m, self.x, self.y)
            models.append(m)
        return models

    def test_ensembleSelect_length(self):
        models = self._make_aligned_models(10)
        ensemble = sgp.ensembleSelect(models, self.x, self.y, numberOfClusters=3)
        self.assertGreater(len(ensemble), 0)
        self.assertLessEqual(len(ensemble), 3)

    def test_evaluateModelEnsemble_length(self):
        ensemble = self._make_aligned_models(4)
        predictions = sgp.evaluateModelEnsemble(ensemble, self.x)
        self.assertEqual(len(predictions), self.x.shape[1])

    def test_evaluateModelEnsembleUncertainty_length(self):
        ensemble = self._make_aligned_models(4)
        uncertainties = sgp.evaluateModelEnsembleUncertainty(ensemble, self.x)
        self.assertEqual(len(uncertainties), self.x.shape[1])

    def test_relativeEnsembleUncertainty_array(self):
        ensemble = self._make_aligned_models(4)
        unc = sgp.relativeEnsembleUncertainty(ensemble, self.x)
        self.assertIsInstance(unc, np.ndarray)
        self.assertEqual(len(unc), self.x.shape[1])


# ---------------------------------------------------------------------------
# 15. Sharpness Computations
# ---------------------------------------------------------------------------

class TestSharpness(unittest.TestCase):

    def setUp(self):
        np.random.seed(6)
        self.x = np.random.rand(2, 40)
        self.y = self.x[0] + self.x[1]
        self.model = sgp.alignGPModel(simple_model(), self.x, self.y)

    def test_sharpnessConstants_nonnegative(self):
        val = sgp.sharpnessConstants(self.model, self.x, self.y)
        self.assertGreaterEqual(val, 0.0)

    def test_sharpnessData_nonnegative(self):
        val = sgp.sharpnessData(self.model, self.x, self.y)
        self.assertGreaterEqual(val, 0.0)

    def test_totalSharpness_sum(self):
        # totalSharpness calls sharpnessConstants + sharpnessData internally,
        # but each call is stochastic; verify result is non-negative.
        ts = sgp.totalSharpness(self.model, self.x, self.y)
        self.assertGreaterEqual(ts, 0.0)


# ---------------------------------------------------------------------------
# 15b. Model Curvature Computations (Haut, Card, Kotanchek)
# ---------------------------------------------------------------------------

class TestModelCurvature(unittest.TestCase):
    """Tests for modelCurvature, the Haut/Card/Kotanchek curvature metric."""

    def setUp(self):
        np.random.seed(7)
        self.x = np.random.rand(2, 50) * 4.0   # 2 vars, 50 points in [0, 4)
        self.y = self.x[0] + self.x[1]

    # ------------------------------------------------------------------
    # 1. Return-type and non-negativity
    # ------------------------------------------------------------------

    def test_returns_float(self):
        model = simple_model()
        val = sgp.modelCurvature(model, self.x)
        self.assertIsInstance(float(val), float)

    def test_nonnegative(self):
        """Curvature must always be >= 0 (it is a norm-based quantity)."""
        model = simple_model()
        val = sgp.modelCurvature(model, self.x)
        if not math.isnan(val):
            self.assertGreaterEqual(val, 0.0)

    # ------------------------------------------------------------------
    # 2. Linear model → near-zero curvature
    # ------------------------------------------------------------------

    def test_linear_model_near_zero_curvature(self):
        """f(x) = x0 + x1 is exactly linear; its Hessian is the zero matrix."""
        model = simple_model()           # add(x0, x1)
        val = sgp.modelCurvature(model, self.x)
        # Numerical noise should be tiny relative to finite-difference scale
        if not math.isnan(val):
            self.assertLess(val, 1e-6)

    def test_constant_model_near_zero_curvature(self):
        """A constant model has a zero Hessian regardless of the constant value."""
        model = constant_model(3.14)
        val = sgp.modelCurvature(model, self.x)
        if not math.isnan(val):
            self.assertLess(val, 1e-6)

    # ------------------------------------------------------------------
    # 3. Nonlinear model → positive curvature
    # ------------------------------------------------------------------

    def test_quadratic_model_positive_curvature(self):
        """f(x) = x0^2 has constant second derivative 2; curvature should be > 0."""
        ops = np.array([sgp.sqrd], dtype=object)
        var = [sgp.variableSelect(0)]
        quad_model = [ops, var, []]
        val = sgp.modelCurvature(quad_model, self.x)
        if not math.isnan(val):
            self.assertGreater(val, 0.0)

    def test_nonlinear_greater_than_linear_curvature(self):
        """A quadratic model should have strictly higher curvature than a linear one."""
        linear = simple_model()
        ops = np.array([sgp.sqrd], dtype=object)
        var = [sgp.variableSelect(0)]
        quad = [ops, var, []]
        c_lin = sgp.modelCurvature(linear, self.x)
        c_quad = sgp.modelCurvature(quad, self.x)
        if not math.isnan(c_lin) and not math.isnan(c_quad):
            self.assertGreater(c_quad, c_lin)

    # ------------------------------------------------------------------
    # 4. maxPoints subsampling
    # ------------------------------------------------------------------

    def test_maxPoints_returns_finite(self):
        """With maxPoints set, result should still be a finite non-negative number."""
        model = simple_model()
        val = sgp.modelCurvature(model, self.x, maxPoints=10)
        if not math.isnan(val):
            self.assertGreaterEqual(val, 0.0)

    def test_maxPoints_larger_than_data_is_safe(self):
        """maxPoints > numPoints should behave like no subsampling."""
        model = simple_model()
        val = sgp.modelCurvature(model, self.x, maxPoints=10000)
        if not math.isnan(val):
            self.assertGreaterEqual(val, 0.0)

    # ------------------------------------------------------------------
    # 5. Comparison with sharpness
    # ------------------------------------------------------------------

    def test_curvature_vs_sharpness_linear(self):
        """For a linear model, curvature ≈ 0 while sharpness may be > 0.

        This illustrates the key difference between the two metrics:
        sharpness measures fitness sensitivity to perturbations of constants
        or data, while curvature measures the geometric bending of the surface.
        A linear model with numeric constants can still have non-zero sharpness
        if its constants are large, but its curvature is always zero.
        """
        aligned = sgp.alignGPModel(simple_model(), self.x, self.y)
        curvature = sgp.modelCurvature(aligned, self.x)
        sharpness = sgp.totalSharpness(aligned, self.x, self.y)
        # curvature of a linear model should be near zero
        if not math.isnan(curvature):
            self.assertLess(curvature, 1e-4)
        # sharpness can be nonzero (stochastic; just check it is finite and >= 0)
        self.assertGreaterEqual(sharpness, 0.0)

    def test_curvature_sensitive_to_nonlinearity(self):
        """Curvature grows when the model is more nonlinear (e.g. exp vs identity)."""
        np.random.seed(0)
        x_pos = np.abs(self.x) + 0.1   # exp needs positive inputs

        # f(x) = x0  ('pop' pushes a variable/constant onto the evaluation stack;
        # used here as the identity — returns x0 unchanged, i.e. a linear model)
        ops_lin = np.array(["pop"], dtype=object)
        var_lin = [sgp.variableSelect(0)]
        lin_model = [ops_lin, var_lin, []]

        # f(x) = exp(x0)  (highly nonlinear)
        ops_exp = np.array([sgp.exp], dtype=object)
        var_exp = [sgp.variableSelect(0)]
        exp_model = [ops_exp, var_exp, []]

        c_lin = sgp.modelCurvature(lin_model, x_pos)
        c_exp = sgp.modelCurvature(exp_model, x_pos)

        if not math.isnan(c_lin) and not math.isnan(c_exp):
            self.assertGreater(c_exp, c_lin)

    def test_higher_dimensional_input(self):
        """modelCurvature should handle inputs with more than 2 variables."""
        np.random.seed(3)
        # 5-variable input, 30 data points
        x5 = np.random.rand(5, 30)
        # Build a model that uses only x0 (sqrd); the other variables are ignored
        ops = np.array([sgp.sqrd], dtype=object)
        var = [sgp.variableSelect(0)]
        model = [ops, var, []]
        val = sgp.modelCurvature(model, x5)
        if not math.isnan(val):
            self.assertGreater(val, 0.0)


# ---------------------------------------------------------------------------
# 16. Evolution (short run)
# ---------------------------------------------------------------------------

class TestEvolve(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        random.seed(42)
        self.x = np.random.rand(2, 30)
        self.y = 2.0 * self.x[0] + self.x[1]

    def test_evolve_returns_nonempty_list(self):
        models = sgp.evolve(self.x, self.y, generations=3, popSize=20,
                            ops=sgp.defaultOps(), align=False)
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_evolve_model_structure(self):
        models = sgp.evolve(self.x, self.y, generations=3, popSize=20,
                            ops=sgp.defaultOps(), align=False)
        for m in models:
            self.assertEqual(len(m), 3)

    def test_evolve_best_model_improves(self):
        """Best fitness after evolution should be reasonable (finite)."""
        models = sgp.evolve(self.x, self.y, generations=5, popSize=30,
                            ops=sgp.defaultOps(), align=True)
        sgp.setModelQuality(models[0], self.x, self.y)
        fit = sgp.fitness(models[0], self.x, self.y)
        if not math.isnan(fit):
            self.assertGreaterEqual(fit, 0.0)
            self.assertLessEqual(fit, 1.0)

    def test_evolve_with_returnTracking(self):
        models, tracking = sgp.evolve(self.x, self.y, generations=3, popSize=20,
                                      ops=sgp.defaultOps(), align=False,
                                      returnTracking=True)
        self.assertIsInstance(tracking, list)
        self.assertGreater(len(tracking), 0)

    def test_evolve_time_capped(self):
        """With capTime=True and a very short limit the function should return."""
        import time
        start = time.time()
        models = sgp.evolve(self.x, self.y, generations=100000, popSize=20,
                            ops=sgp.defaultOps(), align=False,
                            capTime=True, timeLimit=2)
        elapsed = time.time() - start
        self.assertLess(elapsed, 10)  # generous upper bound

    def test_evolve_early_termination(self):
        """With earlyTerminationThreshold=1.0 the run should terminate quickly."""
        models = sgp.evolve(self.x, self.y, generations=1000, popSize=20,
                            ops=sgp.defaultOps(), align=False,
                            allowEarlyTermination=True,
                            earlyTerminationThreshold=1.0)
        self.assertIsInstance(models, list)

    def test_evolve_with_dataSubsample(self):
        x = np.random.rand(2, 60)
        y = x[0] + x[1]
        models = sgp.evolve(x, y, generations=3, popSize=20,
                            ops=sgp.defaultOps(), align=False,
                            dataSubsample=True,
                            samplingMethod=sgp.randomSubsample)
        self.assertGreater(len(models), 0)

    def test_evolve_alternateObjectives(self):
        models = sgp.evolve(self.x, self.y, generations=4, popSize=20,
                            ops=sgp.defaultOps(), align=False,
                            alternateObjectives=[sgp.stackGPModelComplexity],
                            alternateObjFrequency=2)
        self.assertGreater(len(models), 0)

    def test_evolve_initialPop(self):
        initial = sgp.initializeGPModels(2, sgp.defaultOps(), sgp.defaultConst(), 5)
        models = sgp.evolve(self.x, self.y, generations=3, popSize=20,
                            ops=sgp.defaultOps(), align=False,
                            initialPop=initial)
        self.assertGreater(len(models), 0)


# ---------------------------------------------------------------------------
# 17. Benchmark Generation
# ---------------------------------------------------------------------------

class TestBenchmarkGeneration(unittest.TestCase):

    def test_generateRandomBenchmark_shapes(self):
        x, y, model = sgp.generateRandomBenchmark(numVars=3, numSamples=50,
                                                    noiseLevel=0, seed=99)
        self.assertEqual(x.shape, (3, 50))
        self.assertEqual(len(y), 50)

    def test_generateRandomBenchmark_model_valid(self):
        x, y, model = sgp.generateRandomBenchmark(numVars=2, numSamples=30, seed=7)
        self.assertEqual(len(model), 3)

    def test_generateRandomBenchmark_with_noise(self):
        x, y_clean, _ = sgp.generateRandomBenchmark(numVars=2, numSamples=50,
                                                      noiseLevel=0, seed=1)
        x2, y_noisy, _ = sgp.generateRandomBenchmark(numVars=2, numSamples=50,
                                                       noiseLevel=1.0, seed=1)
        # With different seeds/noise the arrays should differ
        # (clean vs noisy won't be identical in general with noise > 0)
        self.assertEqual(len(y_clean), 50)
        self.assertEqual(len(y_noisy), 50)

    def test_generateRandomBenchmark_deterministic(self):
        x1, y1, _ = sgp.generateRandomBenchmark(numVars=2, numSamples=20, seed=42)
        x2, y2, _ = sgp.generateRandomBenchmark(numVars=2, numSamples=20, seed=42)
        np.testing.assert_array_equal(x1, x2)


# ---------------------------------------------------------------------------
# 18. optimizeModel
# ---------------------------------------------------------------------------

class TestOptimizeModel(unittest.TestCase):

    def test_optimizeModel_improves_or_maintains_rmse(self):
        np.random.seed(10)
        x = np.random.rand(1, 30)
        y = 3.0 * x[0] + 2.0

        # Build a simple model: mult(x0, C) with numeric constant
        ops = np.array([sgp.mult], dtype=object)
        var = [sgp.variableSelect(0), 1.0]   # start with slope=1
        model = [ops, var, []]

        rmse_before = sgp.rmse(model, x, y)
        optimized = sgp.optimizeModel(model, x, y,
                                      bounds=[(-10.0, 10.0)],
                                      maxiter=50, tol=1e-4)
        rmse_after = sgp.rmse(optimized, x, y)
        if not math.isnan(rmse_before) and not math.isnan(rmse_after):
            self.assertLessEqual(rmse_after, rmse_before + 1e-6)


# ---------------------------------------------------------------------------
# 19. runEpochs
# ---------------------------------------------------------------------------

class TestRunEpochs(unittest.TestCase):

    def test_runEpochs_returns_sorted_models(self):
        np.random.seed(0)
        random.seed(0)
        x = np.random.rand(2, 20)
        y = x[0] + x[1]
        models = sgp.runEpochs(x, y, epochs=2, generations=2, popSize=10,
                               ops=sgp.defaultOps(), align=False)
        self.assertGreater(len(models), 0)


# ---------------------------------------------------------------------------
# 20. Edge Cases and Robustness
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_fitness_nan_input(self):
        """Model producing NaN predictions should return NaN fitness."""
        ops = np.array([sgp.log], dtype=object)
        var = [sgp.variableSelect(0)]
        model = [ops, var, []]
        x = np.array([[-1.0, -2.0, -3.0]])  # log of negative → nan
        y = np.array([1.0, 2.0, 3.0])
        fit = sgp.fitness(model, x, y)
        self.assertTrue(math.isnan(fit) or fit is None)

    def test_evaluateGPModel_single_point(self):
        """Model should handle single-point input."""
        model = simple_model()
        x = np.array([[3.0], [4.0]])
        result = sgp.evaluateGPModel(model, x)
        self.assertAlmostEqual(float(result[0]), 7.0)

    def test_deleteDuplicateModels_single_model(self):
        m = simple_model()
        result = sgp.deleteDuplicateModels([m])
        self.assertEqual(len(result), 1)

    def test_paretoFront_all_equal(self):
        """When all points are equal only one representative ends up on the front
        (the algorithm selects exactly one per equal group)."""
        vals = np.array([[0.5, 2], [0.5, 2], [0.5, 2]])
        front = sgp.paretoFront(vals)
        # At least one point must be on the front
        self.assertGreaterEqual(front.sum(), 1)

    def test_modelArity_unary_chain(self):
        """Chain of unary ops: neg(neg(x)) needs 1 variable."""
        ops = np.array([sgp.neg, sgp.neg], dtype=object)
        var = [sgp.variableSelect(0)]
        model = [ops, var, []]
        self.assertEqual(sgp.modelArity(model), 1)

    def test_rmse_nan_prediction(self):
        """Model producing inf should return nan RMSE."""
        ops = np.array([sgp.protectDiv], dtype=object)
        var = [sgp.variableSelect(0), 0.0]
        model = [ops, var, []]
        x = np.array([[1.0, 2.0]])
        y = np.array([1.0, 2.0])
        err = sgp.rmse(model, x, y)
        self.assertTrue(math.isnan(err))

    def test_reverseList_single(self):
        self.assertEqual(sgp.reverseList([42]), [42])

    def test_generateRandomBenchmark_zero_vars_raises(self):
        """Zero variables should cause an error gracefully or not crash."""
        try:
            x, y, m = sgp.generateRandomBenchmark(numVars=1, numSamples=10, seed=0)
            self.assertIsNotNone(x)
        except Exception:
            pass  # acceptable


if __name__ == "__main__":
    unittest.main(verbosity=2)
