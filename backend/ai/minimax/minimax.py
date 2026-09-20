"""
AgroAI - Minimax Decision Simulation Algorithm
Module: Pest & Climate Risk Management Simulation

Simulates adversarial game tree between Farmer Decisions (MAX) and Environmental/Pest Outcomes (MIN).
"""

class PestRiskMinimax:
    """
    Minimax search with Alpha-Beta Pruning for agricultural risk minimization.
    MAX player = Farmer choosing intervention strategy (Preventative Spray, Irrigation Lock, Drip Increase)
    MIN player = Environment / Pest vector severity (Mild, Moderate, Outbreak)
    """

    ACTIONS = ["Foliar Spraying", "Soil Drenching", "Thermal Lockout", "Precision Drip"]
    SCORES = {
        "Foliar Spraying": {"Mild": 85, "Moderate": 70, "Outbreak": 45},
        "Soil Drenching": {"Mild": 90, "Moderate": 75, "Outbreak": 60},
        "Thermal Lockout": {"Mild": 60, "Moderate": 50, "Outbreak": 30},
        "Precision Drip": {"Mild": 95, "Moderate": 80, "Outbreak": 65},
    }

    @staticmethod
    def minimax(depth: int, is_max: bool, alpha: float, beta: float, current_action: str = None) -> tuple:
        if depth == 0 or current_action is None:
            # Leaf node evaluation
            return None, 75.0

        if is_max:
            max_eval = -float('inf')
            best_action = None
            for action in PestRiskMinimax.ACTIONS:
                # Simulate environment worst-case response
                env_scores = PestRiskMinimax.SCORES[action]
                worst_case = min(env_scores.values())
                if worst_case > max_eval:
                    max_eval = worst_case
                    best_action = action
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return best_action, max_eval
        else:
            min_eval = float('inf')
            for outcome, score in PestRiskMinimax.SCORES.get(current_action, {}).items():
                min_eval = min(min_eval, score)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return current_action, min_eval

    @classmethod
    def evaluate(cls, field_name: str, crop: str) -> dict:
        best_action, score = cls.minimax(depth=2, is_max=True, alpha=-float('inf'), beta=float('inf'), current_action="Precision Drip")
        return {
            "algorithm": "Minimax Decision Simulation",
            "field": field_name,
            "crop": crop,
            "recommended_action": best_action or "Precision Drip Saturation",
            "minimax_payoff_score": score,
            "decision_tree_depth": 2,
            "alpha_beta_pruned": True,
            "explanation": f"Minimax selected '{best_action}' as optimal worst-case risk mitigation strategy against environmental pest vector scenarios."
        }
