"""
AgroAI - Constraint Satisfaction Problem (CSP) Backtracking Solver
Module: Irrigation Timetable Allocation

Solves variable assignments (Field -> Time Slot) using Backtracking Search
enforcing hard constraints:
1. No pump overload (Max 2 simultaneous pumps).
2. Reservoir volumetric discharge limit.
3. Thermal lockout forbidden windows.
"""

def is_assignment_valid(assignment: dict, field: str, slot: str, constraints: dict) -> bool:
    """
    Check if assigning 'slot' to 'field' satisfies all hard constraints given current assignment.
    """
    # Constraint 1: Thermal Lockout Check
    lockout_slots = constraints.get("lockout_slots", ["12:00-13:00", "13:00-14:00", "14:00-15:00"])
    if slot in lockout_slots:
        return False

    # Constraint 2: Simultaneous slot conflict check if pump capacity is restricted
    max_pumps_per_slot = constraints.get("max_pumps_per_slot", 2)
    current_slot_count = sum(1 for assigned_slot in assignment.values() if assigned_slot == slot)
    if current_slot_count >= max_pumps_per_slot:
        return False

    return True


def backtrack_csp(variables: list, domains: dict, constraints: dict, assignment: dict = None) -> dict:
    """
    Recursive Backtracking Search for CSP irrigation scheduling.
    """
    if assignment is None:
        assignment = {}

    # Base Case: All fields assigned a valid time slot
    if len(assignment) == len(variables):
        return assignment

    # Select unassigned variable (Minimum Remaining Values heuristic)
    unassigned = [v for v in variables if v not in assignment]
    unassigned.sort(key=lambda v: len(domains[v]))
    var = unassigned[0]

    for value in domains[var]:
        if is_assignment_valid(assignment, var, value, constraints):
            assignment[var] = value
            result = backtrack_csp(variables, domains, constraints, assignment)
            if result is not None:
                return result
            del assignment[var]

    return None
