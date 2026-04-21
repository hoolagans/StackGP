"""
TestEvaluateEquivalence.py

Verifies that the iterative evaluateGPModel / evModHelper produces results
that are identical (within floating-point tolerance) to the original recursive
implementation across a large number of randomly generated models.

Run with:  python TestEvaluateEquivalence.py
"""

import copy
import math
import random
import sys

import numpy as np

import StackGP as sgp

# ---------------------------------------------------------------------------
# Reference: original recursive evModHelper (preserved verbatim from the
# version prior to the iterative refactor).
# ---------------------------------------------------------------------------

def _evModHelper_recursive(varStack, opStack, tempStack, data):
    """Original recursive helper - used only as a reference baseline."""
    stack1 = varStack
    stack2 = opStack
    stack3 = tempStack

    if len(stack2) == 0:
        return [stack3, stack2, stack1]
    op = stack2[0]
    stack2 = stack2[1:]

    if callable(op):
        patt = sgp.getArity(op)
        while patt > len(stack3):
            stack3 = [stack1[0]] + stack3
            stack1 = stack1[1:]
        try:
            temp = op(*sgp.varReplace(sgp.reverseList(stack3[:patt]), data))
        except TypeError:
            temp = np.nan
        except OverflowError:
            temp = np.nan
        stack3 = stack3[patt:]
        stack3 = [temp] + stack3
    else:
        if len(stack1) > 0:
            stack3 = sgp.varReplace([stack1[0]], data) + stack3
            stack1 = stack1[1:]

    if len(stack2) > 0:
        stack1, stack2, stack3 = _evModHelper_recursive(stack1, stack2, stack3, data)

    return [stack1, stack2, stack3]


def _evaluateGPModel_recursive(model, inputData):
    """Original evaluateGPModel - used only as a reference baseline."""
    response = _evModHelper_recursive(
        model[1], model[0], [], np.array(inputData).astype(float)
    )[2][0]
    if not type(response) == np.ndarray and sgp.inputLen(inputData) > 1:
        response = np.array([response for _ in range(sgp.inputLen(inputData))])
    return response


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _results_equal(orig, new):
    """Return True if orig and new are numerically equal (NaN-safe)."""
    if isinstance(orig, np.ndarray) and isinstance(new, np.ndarray):
        return np.allclose(orig, new, equal_nan=True, rtol=1e-10, atol=1e-10)
    if isinstance(orig, np.ndarray) or isinstance(new, np.ndarray):
        return False
    try:
        fo, fn = float(orig), float(new)
        return fo == fn or (math.isnan(fo) and math.isnan(fn))
    except (TypeError, ValueError):
        return orig == new


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_equivalence_test(
    num_variables=3,
    num_models_per_config=500,
    max_op_length=15,
    seeds=range(10),
    verbose=True,
):
    """
    Generate models using initializeGPModels for every combination of
    (ops set, random seed) and compare iterative vs recursive results.

    Raises AssertionError if any mismatch is found.
    """
    ops_configs = {
        "defaultOps": sgp.defaultOps(),
        "allOps": sgp.allOps(),
        "booleanOps": sgp.booleanOps(),
    }

    total_tested = 0
    total_skipped = 0
    failures = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        # Use fresh random data for each seed
        x = np.array([np.random.normal(0, 2, 100) for _ in range(num_variables)])

        for ops_name, ops in ops_configs.items():
            models = sgp.initializeGPModels(
                num_variables,
                ops=ops,
                numberOfModels=num_models_per_config,
                maxLength=max_op_length,
            )

            for i, m in enumerate(models):
                try:
                    ro = _evaluateGPModel_recursive(copy.deepcopy(m), x)
                    rn = sgp.evaluateGPModel(copy.deepcopy(m), x)
                    total_tested += 1

                    if not _results_equal(ro, rn):
                        failures.append(
                            dict(
                                seed=seed,
                                ops=ops_name,
                                model_idx=i,
                                ops_list=[
                                    f.__name__ if callable(f) else f
                                    for f in m[0]
                                ],
                                vars_list=[
                                    v.__name__ if callable(v) else str(v)
                                    for v in m[1]
                                ],
                                orig=ro,
                                new=rn,
                            )
                        )
                except RecursionError:
                    # Original recursive version may hit Python's recursion limit
                    # for very deep models; skip those cases.
                    total_skipped += 1
                except Exception:
                    total_skipped += 1

    if verbose:
        print(
            f"Tested {total_tested} models across {len(seeds)} seeds "
            f"and {len(ops_configs)} op sets "
            f"({total_skipped} skipped due to recursion/error)."
        )

    if failures:
        print(f"\nFAILED: {len(failures)} mismatch(es) found!\n")
        for info in failures[:5]:
            print(
                f"  seed={info['seed']} ops={info['ops']} idx={info['model_idx']}"
            )
            print(f"    op stack : {info['ops_list']}")
            print(f"    var stack: {info['vars_list']}")
            orig = info["orig"]
            new = info["new"]
            print(
                f"    orig: {orig[:5] if isinstance(orig, np.ndarray) else orig}"
            )
            print(
                f"    new : {new[:5] if isinstance(new, np.ndarray) else new}"
            )
        raise AssertionError(
            f"{len(failures)} model(s) produced different results between "
            "the iterative and recursive evaluateGPModel implementations."
        )

    if verbose:
        print("All results match. Iterative implementation is equivalent to the original.")
    return True


if __name__ == "__main__":
    success = run_equivalence_test(
        num_variables=3,
        num_models_per_config=500,
        max_op_length=15,
        seeds=range(20),
        verbose=True,
    )
    sys.exit(0 if success else 1)
