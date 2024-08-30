from sympy import symbols, simplify, expand
from StackGP import printGPModel

import signal #for timing out functions
from contextlib import contextmanager #for timing out functions

###################### Timeout function for model complexity ######################
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
####################################################################################

# Counts basis terms in a model
def count_basis_terms(equation, expand=False):
    try:
        with time_limit(2):


            if expand:
                # Simplify the equation to standardize the expression
                simplified_eq = simplify(equation)
                # Expand the expression to identify additive terms clearly
                expanded_eq = expand(simplified_eq)
            
                # Separate the terms of the expression
                terms = expanded_eq.as_ordered_terms()
            else:
                terms = equation.as_ordered_terms()
            #print(terms)
            
    except TimeoutException as e:
        return 1000
    return len(terms)

# Determines the number of basis functions in a model by counting +s and -s
def basisFunctionComplexity(model,*args):
    try:
        return count_basis_terms(printGPModel(model))
    except:
        return 1000

# Creates a lambda function to be used as a complexity metric when given a target dimensionality and deviation
def basisFunctionComplexityDiff(target, deviation):
    return lambda model,*args: min(abs(basisFunctionComplexity(model)-target),(deviation))
