"""Print shot summary to console when a rep completes."""

from feedback.models import ShotSummary


def print_shot_summary(summary: ShotSummary):
    print("\n" + "=" * 50)
    print(f"  SHOT #{summary.shot_number}  —  {summary.grade.upper()}  ({summary.score}/100)")
    print("=" * 50)
    print(f"  Rules: {summary.passed_count}/{summary.total_count} passed")

    if summary.passed_rules:
        print("\n  Passed:")
        for rule in summary.passed_rules:
            print(f"    + {rule.name}")

    if summary.violations:
        print("\n  Fix next:")
        for rule in summary.violations:
            print(f"    - {rule.name}: {rule.message}")

    if summary.coaching_tips:
        print("\n  Coach says:")
        for tip in summary.coaching_tips:
            print(f"    > {tip}")

    print("=" * 50 + "\n")
