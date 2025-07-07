from run_experiments_functions import (
    setup_experiment_directory,
    handle_random_seed,
    setup_cross_validation,
    create_fold_split,
    create_train_test_split,
    process_fold,
    combine_fold_results,
    add_execution_time
)
import argparse
import sys
import os
import time


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-max_k", help="max number of rules in a rule list (def = 5)", type=int, default=5)
    parser.add_argument(
        "-max_z", help="max number of conjunctions in each rule", type=int, default=1)
    parser.add_argument("-results", help="output results path (def. results.csv)",
                        default="results_experiment.csv")
    parser.add_argument("-num_runs", help="number of runs for each experiment",
                        type=int, default=1)
    parser.add_argument("-J", help="number of workers(def. 1)",
                        type=int, default=1)
    parser.add_argument("-minf", help="min freq of conjunctions",
                        type=float, default=0.0)
    parser.add_argument("-v", help="verbose level (def. 1)",
                        type=int, default=0)
    parser.add_argument("-seed", help="random seed (def. None)",
                        type=int, default=None)
    parser.add_argument("-name", help="name of the experiment",
                        type=str)
    parser.add_argument("-db", help="dataset name",
                        type=str)
    parser.add_argument("-constantM", help="Do you want to use constant M? default is 0, 1 for yes",
                        type=int, default=0)
    parser.add_argument("-fold", help="Use 5-fold cross validation (0=no, 1=yes)",
                        type=int, default=0)
    args = parser.parse_args()

    cmd_line = "python3 " + " ".join(sys.argv)

    dataset = args.db
    exp_parameters = [(0.025, 1)]
    max_z = args.max_z
    max_k = args.max_k
    minf = args.minf
    num_runs = args.num_runs
    num_workers = args.J
    use_cv = args.fold == 1
    n_folds = 5 if use_cv else 1

    exp_dir = setup_experiment_directory(args.name, args.v)
    if exp_dir is None:
        return

    seed = handle_random_seed(exp_dir, args.seed, cmd_line, args.v)

    fold_results = []
    base_name = os.path.basename(dataset)
    base_name_without_ext = base_name.replace(".csv", "")

    if use_cv:
        separator, X, y, skf = setup_cross_validation(
            dataset, n_folds, seed, args.v)
        if separator is None or X.empty or y is None or skf is None:
            return

    for fold_idx in range(n_folds):
        if use_cv:
            fold_dir = os.path.join(exp_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            fold_results_dir = os.path.join(fold_dir, "results")
            os.makedirs(fold_results_dir, exist_ok=True)

            fold_train_idx, fold_test_idx = list(skf.split(X, y))[fold_idx]
            train_path, test_path = create_fold_split(
                dataset, exp_dir, fold_idx, fold_train_idx, fold_test_idx, separator, args.v)
        else:
            train_path, test_path = create_train_test_split(
                dataset, exp_dir, test_size=0.2, random_state=seed, verbose=args.v)
            fold_results_dir = os.path.join(exp_dir, "results")
            os.makedirs(fold_results_dir, exist_ok=True)

        fold_results_file = os.path.join(fold_results_dir, args.results)

        fold_result = process_fold(
            fold_idx, dataset, exp_dir, base_name_without_ext,
            train_path, test_path, max_z, minf, fold_results_file,
            exp_parameters, max_k, num_runs, num_workers, seed,
            use_cv, args.constantM, args.v)

        fold_results.append(fold_result)

    if use_cv:
        combine_fold_results(fold_results, exp_dir)

    add_execution_time(exp_dir, start_time)

    print(f"\nExperiment '{args.name}' completed successfully!")


if __name__ == "__main__":
    main()
