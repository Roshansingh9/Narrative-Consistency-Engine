from typing import List, Dict, Tuple
from reasoning.constraints import ConstraintTracker


class BackstoryScorer:
    """Aggregates claim-level judgments into final binary prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.tracker = ConstraintTracker()

        self.hard_weight = config['scoring']['hard_contradiction_weight']
        self.soft_weight = config['scoring']['soft_contradiction_weight']
        self.support_weight = config['scoring']['support_weight']
        self.insufficient_threshold = config['scoring']['insufficient_evidence_threshold']

    def compute_score(self, deliberations: List[Dict]) -> Tuple[int, str]:
        """
        Compute final binary label and rationale.

        Returns: (label, rationale)
            label: 1 = Consistent, 0 = Contradictory
            rationale: Explanation string
        """
        classification = self.tracker.classify_constraints(deliberations)

        # RULE 1: Hard contradictions override everything
        if self.tracker.has_critical_violations(classification):
            rationale = (
                f"CONTRADICTORY: Found {classification['hard_contradictions']} "
                f"hard contradiction(s) that cannot be reconciled with the novel."
            )
            return 0, rationale

        # RULE 2: Insufficient evidence → conservative (CONTRADICTORY)
        coverage = self.tracker.get_evidence_coverage(classification)
        if coverage < (1 - self.insufficient_threshold):
            rationale = (
                f"CONTRADICTORY (conservative): Only {coverage:.0%} of claims "
                f"have sufficient evidence. Insufficient data to validate backstory."
            )
            return 0, rationale

        # RULE 3: Weighted scoring for ambiguous cases
        score = 0.0
        score -= classification['hard_contradictions'] * self.hard_weight
        score -= classification['soft_contradictions'] * self.soft_weight
        score += classification['consistent_claims'] * self.support_weight

        if score >= 0:
            rationale = (
                f"CONSISTENT: {classification['consistent_claims']} claims supported, "
                f"{classification['soft_contradictions']} minor conflicts (resolvable)."
            )
            return 1, rationale
        else:
            rationale = (
                f"CONTRADICTORY: {classification['soft_contradictions']} contradictions "
                f"outweigh {classification['consistent_claims']} supporting claims."
            )
            return 0, rationale

    def score_all(self, all_deliberations: List[List[Dict]]) -> List[Tuple[int, str]]:
        """
        Score multiple backstories.

        Args:
            all_deliberations: List of deliberation lists (one per backstory)

        Returns:
            List of (label, rationale) tuples
        """
        results = []
        for deliberations in all_deliberations:
            label, rationale = self.compute_score(deliberations)
            results.append((label, rationale))

        return results
