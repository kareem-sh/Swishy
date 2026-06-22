"""Generate prioritized coaching text from a shot summary."""

from feedback.models import ShotSummary
from utils.config_loader import load_yaml

_SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1}


def generate_coaching_tips(summary: ShotSummary) -> list[str]:
    """Build ordered coaching tips from violations and overall score."""
    cfg = load_yaml("scoring.yaml")
    max_tips = int(cfg.get("max_coaching_tips", 3))
    thresholds = cfg.get("score_messages", {})

    tips: list[str] = []

    if summary.total_count == 0:
        return summary.coaching_tips or ["Keep your full body visible to the camera."]

    sorted_violations = sorted(
        summary.violations,
        key=lambda r: _SEVERITY_RANK.get(r.severity, 0),
        reverse=True,
    )

    for violation in sorted_violations[:max_tips]:
        tips.append(violation.message)

    score = summary.score
    if score >= int(thresholds.get("excellent", 90)):
        tips.insert(0, thresholds.get("excellent_msg", "Excellent mechanics — keep this rhythm."))
    elif score >= int(thresholds.get("good", 75)):
        tips.append(thresholds.get("good_msg", "Solid form. Refine the notes above."))
    elif score >= int(thresholds.get("fair", 60)):
        tips.append(thresholds.get("fair_msg", "Fair attempt — focus on one correction per rep."))
    else:
        tips.append(thresholds.get("needs_work_msg", "Break the shot into phases and practice each slowly."))

    return tips
