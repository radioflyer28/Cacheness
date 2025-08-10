#!/usr/bin/env python3
"""
Threshold Optimization Benchmark for Cacheness Parallel Hashing

This utility helps find optimal thresholds for when to use parallel vs sequential
directory hashing by measuring performance across different scenarios.

Usage:
    python threshold_benchmark.py

Features:
- Fast file creation using efficient patterns
- Consistent hash verification
- Performance analysis
- Threshold recommendation
"""

import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from cacheness.utils import hash_directory_parallel, _hash_directory_sequential


class ThresholdBenchmark:
    """Benchmark utility for optimizing parallel hashing thresholds."""

    def __init__(self, output_file: str = "threshold_benchmark.json"):
        self.output_file = output_file
        self.results = []

    def create_test_files(
        self, directory: Path, num_files: int, file_size_mb: float
    ) -> None:
        """Create test files efficiently with unique content per file."""
        file_size_bytes = int(file_size_mb * 1024 * 1024)
        base_pattern = b"THRESHOLD_BENCHMARK_"

        for i in range(num_files):
            file_path = directory / f"file_{i:06d}.bin"
            # Create unique pattern for each file to avoid hash collisions
            unique_pattern = base_pattern + f"{i:06d}_".encode() * 8
            pattern_len = len(unique_pattern)

            with open(file_path, "wb") as f:
                written = 0
                while written < file_size_bytes:
                    remaining = file_size_bytes - written
                    chunk = unique_pattern[: min(pattern_len, remaining)]
                    f.write(chunk)
                    written += len(chunk)

    def benchmark_scenario(
        self, num_files: int, file_size_mb: float, runs: int | None = None
    ) -> Dict[str, Any]:
        """Benchmark a specific file count and size scenario."""
        total_size_mb = num_files * file_size_mb
        description = f"{num_files} files × {file_size_mb}MB = {total_size_mb}MB total"

        # Adjust runs based on scenario size to keep benchmark reasonable
        if runs is None:
            if total_size_mb < 1000:  # < 1GB
                runs = 3
            elif total_size_mb < 5000:  # < 5GB
                runs = 2
            else:  # >= 5GB
                runs = 1

        print(f"\n📊 Testing: {description} (runs: {runs})")

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = Path(tmp_dir) / "test_data"
            test_dir.mkdir()

            # Create test files
            print(f"   📁 Creating {num_files} files...")
            start = time.time()
            self.create_test_files(test_dir, num_files, file_size_mb)
            creation_time = time.time() - start
            print(f"   ⏱️  File creation: {creation_time:.2f}s")

            # Prepare file list
            file_paths = [(f, test_dir) for f in test_dir.rglob("*") if f.is_file()]

            # Benchmark sequential
            print("   🔄 Testing sequential...")
            seq_times = []
            for run in range(runs):
                start = time.time()
                seq_hash = _hash_directory_sequential(test_dir, file_paths)
                duration = time.time() - start
                seq_times.append(duration)
                if runs > 1:
                    print(f"      Run {run + 1}: {duration:.3f}s")

            # Benchmark parallel
            print("   ⚡ Testing parallel...")
            par_times = []
            for run in range(runs):
                start = time.time()
                par_hash = hash_directory_parallel(test_dir, max_workers=None)
                duration = time.time() - start
                par_times.append(duration)
                if runs > 1:
                    print(f"      Run {run + 1}: {duration:.3f}s")

            # Calculate metrics
            avg_seq = statistics.mean(seq_times)
            avg_par = statistics.mean(par_times)
            speedup = avg_seq / avg_par if avg_par > 0 else 0
            overhead = avg_par - avg_seq

            # Determine current threshold decision
            current_would_parallel = total_size_mb > 4000 or num_files > 80

            result = {
                "num_files": num_files,
                "file_size_mb": file_size_mb,
                "total_size_mb": total_size_mb,
                "creation_time": creation_time,
                "sequential_times": seq_times,
                "parallel_times": par_times,
                "avg_sequential": avg_seq,
                "avg_parallel": avg_par,
                "speedup": speedup,
                "overhead": overhead,
                "hash_consistent": seq_hash == par_hash,
                "current_threshold_parallel": current_would_parallel,
                "beneficial": speedup > 1.1,  # 10% improvement threshold
                "description": description,
                "runs": runs,
            }

            print(f"   📊 Sequential: {avg_seq:.3f}s, Parallel: {avg_par:.3f}s")
            print(
                f"   🚀 Speedup: {speedup:.2f}x, Beneficial: {'✅' if result['beneficial'] else '❌'}"
            )
            print(
                f"   🔍 Hash consistent: {'✅' if result['hash_consistent'] else '❌'}"
            )

            if not result["hash_consistent"]:
                print("   ⚠️  Hash mismatch detected!")

            return result

    def run_comprehensive_benchmark(self) -> None:
        """Run a comprehensive benchmark to find optimal thresholds."""
        print("🚀 THRESHOLD OPTIMIZATION BENCHMARK")
        print("=" * 50)
        print("Finding optimal thresholds for parallel vs sequential hashing...")

        # Test scenarios covering a much wider range to find true thresholds
        scenarios = [
            # Small scenarios (definitely sequential)
            (5, 1),  # 5MB total
            (10, 2),  # 20MB total
            (20, 5),  # 100MB total
            (50, 2),  # 100MB total, many files
            # Medium scenarios (current threshold boundary)
            (25, 10),  # 250MB total
            (50, 10),  # 500MB total (current size threshold)
            (100, 5),  # 500MB total, many files (current file threshold)
            (200, 2),  # 400MB total, very many small files
            # Large scenarios (above current thresholds)
            (100, 10),  # 1GB total
            (200, 10),  # 2GB total
            (150, 20),  # 3GB total
            (100, 30),  # 3GB total, fewer larger files
            (300, 10),  # 3GB total, many medium files
            (500, 5),  # 2.5GB total, very many smaller files
            # Very large scenarios (should definitely benefit)
            (200, 25),  # 5GB total
            (150, 50),  # 7.5GB total
            (100, 75),  # 7.5GB total, fewer larger files
            (300, 25),  # 7.5GB total, many medium files
            (500, 15),  # 7.5GB total, very many files
            # Extreme scenarios (to find upper limits)
            (200, 50),  # 10GB total
            (100, 100),  # 10GB total, fewer very large files
            (500, 20),  # 10GB total, many files
            (1000, 10),  # 10GB total, very many files
            # Massive scenarios (if system can handle)
            (300, 50),  # 15GB total
            (200, 75),  # 15GB total
            (500, 30),  # 15GB total, many files
            (1000, 15),  # 15GB total, very many files
        ]

        print(f"Running {len(scenarios)} scenarios (from small to massive)...")
        print("💡 Tip: Press Ctrl+C at any time to stop and analyze current results")
        start_time = time.time()

        for i, (num_files, file_size_mb) in enumerate(scenarios, 1):
            try:
                print(f"\n{'=' * 60}")
                print(f"🎯 SCENARIO {i}/{len(scenarios)}")

                result = self.benchmark_scenario(num_files, file_size_mb)
                self.results.append(result)

                # Show running summary
                beneficial_count = sum(1 for r in self.results if r["beneficial"])
                print(
                    f"📈 Running summary: {beneficial_count}/{len(self.results)} scenarios beneficial so far"
                )

            except KeyboardInterrupt:
                print(f"\n⚠️  Benchmark interrupted by user after {i - 1} scenarios")
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print("   Continuing to next scenario...")
                continue

        total_time = time.time() - start_time
        print(f"\n🏁 Completed in {total_time:.1f} seconds")

        # Save and analyze results
        self.save_results()
        self.analyze_and_recommend()

    def save_results(self) -> None:
        """Save benchmark results."""
        data = {
            "timestamp": time.time(),
            "results": self.results,
            "system_info": {
                "total_scenarios": len(self.results),
                "benchmark_version": "1.0",
            },
        }

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Results saved to {self.output_file}")

    def analyze_and_recommend(self) -> None:
        """Analyze results and recommend optimal thresholds."""
        if not self.results:
            print("❌ No results to analyze")
            return

        print("\n📊 ANALYSIS & RECOMMENDATIONS")
        print("=" * 40)

        # Check hash consistency
        inconsistent = [r for r in self.results if not r["hash_consistent"]]
        if inconsistent:
            print(f"⚠️  {len(inconsistent)} scenarios had hash inconsistencies!")
            return

        # Find beneficial scenarios
        beneficial = [r for r in self.results if r["beneficial"]]
        not_beneficial = [r for r in self.results if not r["beneficial"]]

        print(f"✅ Scenarios where parallel is beneficial: {len(beneficial)}")
        print(f"❌ Scenarios where sequential is better: {len(not_beneficial)}")

        if beneficial:
            print("\n🚀 PARALLEL BENEFICIAL SCENARIOS:")
            for r in beneficial:
                print(f"   {r['description']} - {r['speedup']:.2f}x speedup")

            # Find minimum thresholds for beneficial scenarios
            min_total_size = min(r["total_size_mb"] for r in beneficial)
            min_file_count = min(r["num_files"] for r in beneficial)
            avg_speedup = statistics.mean(r["speedup"] for r in beneficial)

            print("\n💡 RECOMMENDATIONS:")
            print(f"   • Minimum beneficial total size: {min_total_size:.0f}MB")
            print(f"   • Minimum beneficial file count: {min_file_count}")
            print(f"   • Average speedup when beneficial: {avg_speedup:.2f}x")

        if not_beneficial:
            avg_overhead = statistics.mean(r["overhead"] for r in not_beneficial)
            print(f"\n⚠️  Average overhead when not beneficial: {avg_overhead:.3f}s")

        # Evaluate current thresholds
        current_correct = 0
        false_positives = 0
        false_negatives = 0

        for r in self.results:
            predicted = r["current_threshold_parallel"]
            actual = r["beneficial"]

            if predicted == actual:
                current_correct += 1
            elif predicted and not actual:
                false_positives += 1
            elif not predicted and actual:
                false_negatives += 1

        accuracy = current_correct / len(self.results) * 100
        print(f"\n🎯 CURRENT THRESHOLD ACCURACY: {accuracy:.1f}%")
        print(f"   False positives (unnecessary parallel): {false_positives}")
        print(f"   False negatives (missed opportunities): {false_negatives}")

        if accuracy < 80:
            print("\n🔧 THRESHOLD ADJUSTMENT NEEDED")
            if beneficial:
                suggested_size = min(r["total_size_mb"] for r in beneficial) * 0.8
                suggested_files = min(r["num_files"] for r in beneficial) * 0.8
                print("   Suggested thresholds:")
                print(f"   • Size threshold: {suggested_size:.0f}MB")
                print(f"   • File count threshold: {suggested_files:.0f}")
        else:
            print("✅ Current thresholds are working well!")


def main():
    """Run the threshold optimization benchmark."""
    print("🎯 Cacheness Parallel Hashing Threshold Optimizer")
    print("This will test various scenarios to find optimal thresholds.")
    print("Press Ctrl+C to stop early if needed.\n")

    benchmark = ThresholdBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
