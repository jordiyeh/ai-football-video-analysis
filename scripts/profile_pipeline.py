#!/usr/bin/env python3
"""Profile the video analysis pipeline using cProfile or py-spy.

Usage:
    # Profile with cProfile (built-in, no extra deps)
    python scripts/profile_pipeline.py --video match.mp4 --output runs/profile --mode cprofile

    # Generate flame graph with py-spy (requires: pip install py-spy)
    python scripts/profile_pipeline.py --video match.mp4 --output runs/profile --mode flamegraph

    # Analyze existing profile data
    python scripts/profile_pipeline.py --analyze runs/profile/profile.prof

    # Quick summary of existing profile
    python scripts/profile_pipeline.py --analyze runs/profile/profile.prof --top 20
"""

import argparse
import cProfile
import io
import pstats
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cprofile(video_path: str, output_dir: str, extra_args: list[str] | None = None) -> Path:
    """Run the pipeline with cProfile instrumentation.

    Args:
        video_path: Path to video file
        output_dir: Output directory
        extra_args: Extra arguments for the CLI

    Returns:
        Path to the generated .prof file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    profile_path = output_path / "profile.prof"

    # Build the import and run commands
    profiler = cProfile.Profile()

    # Import the CLI module and run it
    print(f"Profiling pipeline on: {video_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        # Prepare sys.argv for Click
        sys.argv = [
            "cli.py",
            "--video", video_path,
            "--output", output_dir,
        ]
        if extra_args:
            sys.argv.extend(extra_args)

        # Import here to avoid circular imports
        from src.cli import main

        # Profile the main function
        profiler.enable()
        try:
            main(standalone_mode=False)
        except SystemExit:
            pass  # Click raises SystemExit on completion
        profiler.disable()

    except Exception as e:
        print(f"Error during profiling: {e}")
        raise

    # Save profile data
    profiler.dump_stats(str(profile_path))
    print(f"\nProfile data saved to: {profile_path}")

    # Generate summary
    summary_path = output_path / "profile_summary.txt"
    generate_profile_summary(profile_path, summary_path)

    return profile_path


def run_pyspy(video_path: str, output_dir: str, extra_args: list[str] | None = None) -> Path:
    """Run the pipeline with py-spy for flame graph generation.

    Args:
        video_path: Path to video file
        output_dir: Output directory
        extra_args: Extra arguments for the CLI

    Returns:
        Path to the generated SVG flame graph
    """
    # Check if py-spy is available
    if shutil.which("py-spy") is None:
        print("Error: py-spy not found. Install with: pip install py-spy")
        print("Note: On macOS, you may need to run with sudo or disable SIP")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    svg_path = output_path / "flamegraph.svg"
    speedscope_path = output_path / "profile.speedscope.json"

    # Build the CLI command
    cli_args = [
        sys.executable,
        "src/cli.py",
        "--video", video_path,
        "--output", output_dir,
    ]
    if extra_args:
        cli_args.extend(extra_args)

    print(f"Profiling pipeline with py-spy on: {video_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Run py-spy record
    pyspy_cmd = [
        "py-spy", "record",
        "-o", str(svg_path),
        "--format", "speedscope",
        "-o", str(speedscope_path),
        "--",
    ] + cli_args

    # Try without sudo first
    try:
        subprocess.run(pyspy_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"py-spy failed: {e}")
        print("Try running with sudo if on macOS")
        raise

    print(f"\nFlame graph saved to: {svg_path}")
    print(f"Speedscope profile saved to: {speedscope_path}")
    print(f"View in browser: https://www.speedscope.app/ (load {speedscope_path})")

    return svg_path


def generate_profile_summary(profile_path: Path, output_path: Path, top_n: int = 50) -> None:
    """Generate a human-readable summary of profile data.

    Args:
        profile_path: Path to .prof file
        output_path: Path to save summary text
        top_n: Number of top functions to include
    """
    stats = pstats.Stats(str(profile_path))

    # Capture output to string
    output = io.StringIO()

    output.write(f"Profile Summary - Generated {datetime.now().isoformat()}\n")
    output.write("=" * 80 + "\n\n")

    # Top functions by cumulative time
    output.write(f"TOP {top_n} FUNCTIONS BY CUMULATIVE TIME\n")
    output.write("-" * 80 + "\n\n")

    stream = io.StringIO()
    stats.stream = stream
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    output.write(stream.getvalue())

    output.write("\n\n")

    # Top functions by total time (self time)
    output.write(f"TOP {top_n} FUNCTIONS BY TOTAL (SELF) TIME\n")
    output.write("-" * 80 + "\n\n")

    stream = io.StringIO()
    stats.stream = stream
    stats.sort_stats("tottime")
    stats.print_stats(top_n)
    output.write(stream.getvalue())

    output.write("\n\n")

    # Callers of key functions (optional analysis)
    output.write("OPTIMIZATION OPPORTUNITIES\n")
    output.write("-" * 80 + "\n\n")

    # Get stats data for analysis
    stats_data = stats.stats

    # Find functions that are called many times with high total time
    hot_functions = []
    for func_key, (cc, nc, tt, ct, callers) in stats_data.items():
        filename, line, func_name = func_key
        if nc > 1000 and tt > 0.1:  # Called >1000 times with >0.1s total
            hot_functions.append({
                "name": f"{func_name} ({filename}:{line})",
                "calls": nc,
                "total_time": tt,
                "per_call": tt / nc if nc > 0 else 0,
            })

    hot_functions.sort(key=lambda x: x["total_time"], reverse=True)

    if hot_functions:
        output.write("Hot functions (called >1000 times with >0.1s total time):\n\n")
        for hf in hot_functions[:20]:
            output.write(f"  {hf['name']}\n")
            output.write(f"    Calls: {hf['calls']:,}, Total: {hf['total_time']:.3f}s, ")
            output.write(f"Per call: {hf['per_call']*1000:.3f}ms\n\n")
    else:
        output.write("No obvious hot spots found.\n\n")

    # Write to file
    with open(output_path, "w") as f:
        f.write(output.getvalue())

    print(f"Summary saved to: {output_path}")


def analyze_profile(profile_path: str, top_n: int = 30) -> None:
    """Analyze an existing profile and print summary.

    Args:
        profile_path: Path to .prof file
        top_n: Number of top functions to show
    """
    profile_path = Path(profile_path)

    if not profile_path.exists():
        print(f"Error: Profile file not found: {profile_path}")
        sys.exit(1)

    print(f"Analyzing profile: {profile_path}\n")

    stats = pstats.Stats(str(profile_path))

    print(f"TOP {top_n} FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)

    print(f"\nTOP {top_n} FUNCTIONS BY TOTAL (SELF) TIME")
    print("=" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(top_n)

    # Generate summary file alongside the profile
    summary_path = profile_path.parent / "profile_summary.txt"
    generate_profile_summary(profile_path, summary_path, top_n * 2)


def main():
    parser = argparse.ArgumentParser(
        description="Profile the video analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Profiling mode
    parser.add_argument(
        "--mode",
        choices=["cprofile", "flamegraph"],
        default="cprofile",
        help="Profiling mode: cprofile (built-in) or flamegraph (requires py-spy)",
    )

    # Input/output
    parser.add_argument(
        "--video",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        help="Output directory for profile data and results",
    )

    # Analysis mode
    parser.add_argument(
        "--analyze",
        metavar="PROFILE_FILE",
        help="Analyze an existing .prof file instead of running profiling",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top functions to display (default: 30)",
    )

    # Extra CLI arguments
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip overlay video generation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing outputs",
    )

    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        analyze_profile(args.analyze, args.top)
        return

    # Profiling mode - require video and output
    if not args.video or not args.output:
        parser.error("--video and --output are required for profiling mode")

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Build extra args
    extra_args = []
    if args.no_overlay:
        extra_args.append("--no-overlay")
    if args.resume:
        extra_args.append("--resume")

    # Run profiling
    if args.mode == "cprofile":
        run_cprofile(args.video, args.output, extra_args)
    elif args.mode == "flamegraph":
        run_pyspy(args.video, args.output, extra_args)


if __name__ == "__main__":
    main()
