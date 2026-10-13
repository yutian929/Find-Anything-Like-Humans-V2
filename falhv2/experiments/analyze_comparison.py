"""Sensitivity of route ranking to an explicitly hypothetical stop-and-turn model."""
import json
import math
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parents[1] / "outputs/coverage_experiment"
    comparison = json.loads((folder / "comparison.json").read_text())
    rows = []
    wins = {name: 0 for name in comparison["cases"][0]["algorithms"]}
    for case in comparison["cases"]:
        for speed in (0.2, 0.4, 0.6):
            for omega in (0.4, 0.8, 1.2):
                times = {name: data["path_length_m"]/speed + data["quarter_turns"]*math.pi/2/omega
                         for name, data in case["algorithms"].items() if data["status"] == "complete"}
                winner = min(times, key=times.get)
                wins[winner] += 1
                rows.append({"case": case["name"], "linear_speed_m_s": speed, "angular_speed_rad_s": omega,
                             "modeled_seconds": times, "winner": winner})
    result = {"model": "length / linear_speed + quarter_turns * pi/2 / angular_speed",
              "not_measured_robot_execution": True,
              "excludes": ["acceleration", "local navigation behavior", "sensing", "motion between disconnected regions"],
              "win_counts": wins, "scenarios": rows}
    (folder / "sensitivity.json").write_text(json.dumps(result, indent=2)+"\n")
    print(wins)


if __name__ == "__main__":
    main()
