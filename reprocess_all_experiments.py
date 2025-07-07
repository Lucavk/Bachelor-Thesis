import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess experiment results keeping all duplicates")
    parser.add_argument("-d", "--directory", default="results_expiriment",
                        help="Base directory containing experiment results (default: results_expiriment)")
    parser.add_argument("-j", "--jobs", type=int, default=6,
                        help="Number of parallel jobs for processing (default: 6)")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Verbose level (0-2) (default: 1)")
    args = parser.parse_args()

    results_files = []
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file == "results_experiment.csv":
                results_files.append(os.path.join(root, file))

    if args.verbose > 0:
        print(
            f"Found {len(results_files)} results_experiment.csv files to process")

    for i, results_file in enumerate(results_files):
        folder_path = os.path.dirname(results_file)
        performance_file = os.path.join(
            folder_path, "results_performance_dupl.csv")

        if args.verbose > 0:
            print(f"\n[{i+1}/{len(results_files)}] Processing {results_file}")

        cmd = f"python process_output_csv.py -fn {results_file} -o {performance_file} -j {args.jobs} -exp_dir {folder_path} -v {args.verbose} -keep_dupl 1"

        if args.verbose > 0:
            print(f"Running: {cmd}")

        subprocess.run(cmd, shell=True)

        if args.verbose > 0:
            print(f"Finished processing {results_file}")
            print(f"Output saved to {performance_file}")

    if args.verbose > 0:
        print(
            f"\nCompleted reprocessing {len(results_files)} experiment files with duplicates preserved")


if __name__ == "__main__":
    main()
