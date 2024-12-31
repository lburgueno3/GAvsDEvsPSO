import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple

class ConstraintType:
    EQUALITY = "eq"
    INEQUALITY = "ineq"
    DOUBLE_SIDED = "double"

def create_constraint_checker(constraint_func: Callable, constraint_type: str, tolerance: float = 1e-6):
    """
    Creates a standardized constraint checker that returns violation amount
    """
    def check_inequality(x):
        return max(0, -constraint_func(x))  # for inequalities where constraint >= 0
    
    def check_equality(x):
        return abs(constraint_func(x))  # for equalities where constraint = 0
    
    def check_double_sided(x):
        # For constraints in the form form a <= constraint <= b
        result = constraint_func(x)
        if isinstance(result, tuple):
            lower, upper = result[0], result[1]
            val = result[2]
            return max(0, lower - val, val - upper)
        return 0
    
    if constraint_type == ConstraintType.INEQUALITY:
        return check_inequality
    elif constraint_type == ConstraintType.EQUALITY:
        return check_equality
    else:  
        return check_double_sided