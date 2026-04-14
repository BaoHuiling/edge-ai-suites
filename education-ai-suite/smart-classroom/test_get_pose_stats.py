"""
Test script for get_pose_stats using recorded front_posture.txt
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from components.va.va_pipeline_service import VideoAnalyticsPipelineService

FRONT_POSTURE_FILE = (
    "storage/smart-classroom/20260408-123343-b10f/va/front_posture.txt"
)


def run_single_pass():
    """Process the entire file in one call and print stats."""
    print("=" * 60)
    print("Single-pass: processing full file")
    print("=" * 60)

    svc = VideoAnalyticsPipelineService()
    t0 = time.perf_counter()
    stats, state = svc.get_pose_stats(FRONT_POSTURE_FILE)
    elapsed = time.perf_counter() - t0

    print(f"  student_count  : {stats['student_count']}")
    print(f"  stand_count    : {stats['stand_count']}")
    print(f"  raise_up_count : {stats['raise_up_count']}")
    print(f"  stand_reid     : {stats['stand_reid']}")
    print(f"  frames parsed  : {state['total_frames']}")
    print(f"  elapsed        : {elapsed:.2f}s")
    return stats, state


def run_incremental(chunk_size: int = 10_000):
    """
    Simulate incremental processing by calling get_pose_stats in chunks.
    Mimics the live-pipeline scenario where new lines arrive continuously.
    """
    print()
    print("=" * 60)
    print(f"Incremental pass: chunk_size={chunk_size} lines")
    print("=" * 60)

    posture_path = Path(FRONT_POSTURE_FILE)
    all_lines = posture_path.read_text(encoding="utf-8").splitlines(keepends=True)
    total_lines = len(all_lines)
    print(f"  Total lines in file: {total_lines}")

    tmp_path = posture_path.parent / "_test_incremental_posture.txt"
    try:
        svc = VideoAnalyticsPipelineService()
        state = None

        for chunk_start in range(0, total_lines, chunk_size):
            chunk = all_lines[chunk_start : chunk_start + chunk_size]
            with open(tmp_path, "a", encoding="utf-8") as f:
                f.writelines(chunk)
            stats, state = svc.get_pose_stats(str(tmp_path), state)

        print(f"  student_count  : {stats['student_count']}")
        print(f"  stand_count    : {stats['stand_count']}")
        print(f"  raise_up_count : {stats['raise_up_count']}")
        print(f"  stand_reid     : {stats['stand_reid']}")
        print(f"  frames parsed  : {state['total_frames']}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return stats, state


def compare_results(stats_single, stats_incremental):
    print()
    print("=" * 60)
    print("Comparison: single-pass vs incremental")
    print("=" * 60)
    keys = ["student_count", "stand_count", "raise_up_count"]
    all_match = True
    for k in keys:
        match = stats_single[k] == stats_incremental[k]
        status = "OK" if match else "MISMATCH"
        print(f"  {k:20s}: {stats_single[k]} vs {stats_incremental[k]}  [{status}]")
        if not match:
            all_match = False
    print()
    if all_match:
        print("  All key stats match.")
    else:
        print("  WARNING: some stats differ between passes.")


if __name__ == "__main__":
    stats_single, _ = run_single_pass()
    stats_incremental, _ = run_incremental(chunk_size=10_000)
    compare_results(stats_single, stats_incremental)
