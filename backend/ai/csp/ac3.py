"""
AgroAI - Constraint Satisfaction Problem (CSP) & AC-3 Algorithm
Module: Irrigation Scheduling Domain Reduction

Educational Concept:
- Variable: A component needing an assignment (e.g. Field_A, Field_B)
- Domain: Set of possible values for a variable (e.g. Morning, Afternoon, Evening)
- Constraint: Rule specifying allowable combinations (e.g. Pump cannot serve two fields at same time)
- Arc Consistency (AC-3): Reduces domains by removing values that have no valid partner satisfying constraints.
"""

def is_consistent(x_val: str, y_val: str) -> bool:
    """
    Check if assignment (X = x_val) and (Y = y_val) satisfies constraint.
    Constraint: Fields cannot share the exact same irrigation time slot if pump capacity is 1.
    """
    return x_val != y_val


def revise(domains: dict, X: str, Y: str) -> bool:
    """
    Revise domain of X with respect to Y.
    Removes values from X's domain that have no compatible value in Y's domain.
    """
    revised = False
    # Copy domain list to safely iterate while removing elements
    for x_val in list(domains[X]):
        # Check if there exists ANY value y in Y's domain satisfying constraint
        has_compatible_y = any(is_consistent(x_val, y_val) for y_val in domains[Y])
        if not has_compatible_y:
            domains[X].remove(x_val)
            revised = True
    return revised


def ac3(variables: list, domains: dict) -> bool:
    """
    AC-3 (Arc Consistency Algorithm #3)
    Enforces arc consistency across all pairwise variable constraints.
    Returns False if an empty domain is encountered (inconsistent), True otherwise.
    """
    # Create arc queue of all directed pairs (X, Y) where X != Y
    queue = [(X, Y) for X in variables for Y in variables if X != Y]

    while queue:
        (X, Y) = queue.pop(0)

        # Remove inconsistent values from domain of X
        if revise(domains, X, Y):
            # If domain of X becomes empty, no solution exists under constraints
            if len(domains[X]) == 0:
                return False

            # Re-check arcs pointing to X because X's domain shrunk
            for Z in variables:
                if Z != X and Z != Y:
                    queue.append((Z, X))

    return True


if __name__ == "__main__":
    print("=== AgroAI Educational AC-3 Irrigation Solver Demonstration ===")

    # Defined AgroAI Irrigation Variables
    variables = ["Field_A", "Field_B", "Field_C"]

    # Initial Domains for time slots
    # Field_A can only irrigate in Morning or Evening
    # Field_B can only irrigate in Morning
    # Field_C can irrigate Morning, Afternoon, or Evening
    sample_domains = {
        "Field_A": ["Morning", "Evening"],
        "Field_B": ["Morning"],
        "Field_C": ["Morning", "Afternoon", "Evening"]
    }

    print("\n[1] Initial Variable Domains:")
    for var, dom in sample_domains.items():
        print(f"  - {var}: {dom}")

    print("\n[2] Executing AC-3 Arc Consistency Algorithm...")
    success = ac3(variables, sample_domains)

    if success:
        print("\n[3] Domain Reduction Result (Arc Consistent):")
        for var, dom in sample_domains.items():
            print(f"  - {var}: {dom}")
        print("\nEducational Note: AC-3 successfully eliminated conflicting slots before search!")
    else:
        print("\n[!] Inconsistent constraints: No valid schedule possible.")
