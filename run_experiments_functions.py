import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from multiprocessing import Pool
import time


def execute_cmd(cmd):
    """Execute a shell command and print it first."""
    print(cmd)
    os.system(cmd)


def add_execution_time(exp_dir, start_time):
    """
    Add execution time to the random_seed.txt file
    """
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60

    time_msg = f"\nTotal execution time: {execution_time_seconds:.2f} seconds ({execution_time_minutes:.2f} minutes)"

    random_seed_file = os.path.join(exp_dir, "random_seed.txt")

    if os.path.exists(random_seed_file):
        with open(random_seed_file, 'a') as f:
            f.write(time_msg)

    print(time_msg)


def create_train_test_split(input_file, output_dir, test_size=0.2, random_state=None, verbose=0):
    """
    Create train/test split for a dataset and save in output directory
    """
    print(f"Creating train/test split for {input_file}")

    try:
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.count(';') > first_line.count(','):
                sep = ';'
            else:
                sep = ','
    except Exception as e:
        print(f"Error reading file: {e}")
        sep = ','

    try:
        df = pd.read_csv(input_file, sep=sep)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

    if df.empty or df.isna().all().all():
        print(
            f"ERROR: Failed to properly load {input_file}. Check file format.")
        return None, None

    if verbose > 0:
        print(f"Loaded dataset shape: {df.shape}")

    if '{T}' in df.columns:
        y = df['{T}']
        X = df.drop('{T}', axis=1)
    else:
        first_col = df.columns[0]
        y = df[first_col]
        X = df.drop(first_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    label_col = '{T}' if '{T}' in df.columns else df.columns[0]
    train_df = pd.concat(
        [pd.DataFrame(y_train, columns=[label_col]), X_train], axis=1)
    test_df = pd.concat(
        [pd.DataFrame(y_test, columns=[label_col]), X_test], axis=1)

    if verbose > 0:
        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")

    if train_df.empty or test_df.empty:
        print("WARNING: One or both dataframes are empty!")
        return None, None

    train_missing = train_df.isna().sum().sum()
    test_missing = test_df.isna().sum().sum()
    if verbose > 0:
        print(f"Missing values in train set: {train_missing}")
        print(f"Missing values in test set: {test_missing}")

    base_name = os.path.basename(input_file).replace(".csv", "")
    train_path = os.path.join(output_dir, f"{base_name}_train.csv")
    test_path = os.path.join(output_dir, f"{base_name}_test.csv")

    train_df.to_csv(train_path, index=False, sep=sep)
    test_df.to_csv(test_path, index=False, sep=sep)
    if verbose > 0:
        print(f"Saved train set ({len(train_df)} rows) to {train_path}")
        print(f"Saved test set ({len(test_df)} rows) to {test_path}")

    return train_path, test_path


def create_fold_split(input_file, output_dir, fold_idx, train_indices, test_indices, separator=';', verbose=0):
    """
    Create train/test split for a specific fold and save in output directory
    """
    print(f"Creating fold {fold_idx} split for {input_file}")

    try:
        df = pd.read_csv(input_file, sep=separator)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    base_name = os.path.basename(input_file).replace(".csv", "")
    train_path = os.path.join(fold_dir, f"{base_name}_train.csv")
    test_path = os.path.join(fold_dir, f"{base_name}_test.csv")

    train_df.to_csv(train_path, index=False, sep=separator)
    test_df.to_csv(test_path, index=False, sep=separator)
    if verbose > 0:
        print(
            f"Saved fold {fold_idx} train set ({len(train_df)} rows) to {train_path}")
        print(
            f"Saved fold {fold_idx} test set ({len(test_df)} rows) to {test_path}")

    return train_path, test_path


def setup_experiment_directory(name, verbose=0):
    """
    Set up experiment directory structure.
    Returns the path to the experiment directory.
    """
    if not os.path.exists("results_experiments"):
        os.makedirs("results_experiments")

    exp_dir = os.path.join("results_experiments", name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        if verbose > 0:
            print(f"Created directory: {exp_dir}")
    else:
        if verbose > 0:
            print(
                f"Directory {exp_dir} already exists. Please remove it or choose a different name.")
            return None

    return exp_dir


def handle_random_seed(exp_dir, seed=None, cmd_line="", verbose=0):
    """
    Handle random seed generation or use provided seed.
    Returns the seed value.
    """
    if seed is None:
        seed = random.randint(0, 1000000)
        seed_file_path = os.path.join(exp_dir, "random_seed.txt")
        with open(seed_file_path, "w") as seed_file:
            seed_file.write(f"Random seed used: {seed}\n")
            seed_file.write(
                f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            seed_file.write(
                "This seed can be used with the -seed parameter to reproduce this experiment exactly.\n")
            if 'seed' not in cmd_line:
                cmd_line += f" -seed {seed}"
            seed_file.write(
                f"\nCommand used to run this experiment:\n{cmd_line}\n")
        if verbose > 0:
            print(f"Generated random seed: {seed}")
            print(f"Seed saved to: {seed_file_path}")
    else:
        seed_file_path = os.path.join(exp_dir, "random_seed.txt")
        with open(seed_file_path, "w") as seed_file:
            seed_file.write(f"User-provided seed: {seed}\n")
            seed_file.write(
                f"Experiment run on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if 'seed' not in cmd_line:
                cmd_line += f" -seed {seed}"
            seed_file.write(
                f"\nCommand used to run this experiment:\n{cmd_line}\n")

    random.seed(seed)
    return seed


def setup_cross_validation(dataset, n_folds, seed, verbose=0):
    """
    Set up cross-validation splits.
    Returns separator, X, y, and the stratified k-fold splitter.
    """
    try:
        with open(dataset, 'r') as f:
            first_line = f.readline().strip()
            separator = ';' if first_line.count(
                ';') > first_line.count(',') else ','

        df = pd.read_csv(dataset, sep=separator)

        if '{T}' in df.columns:
            y = df['{T}']
            X = df.drop('{T}', axis=1)
        else:
            first_col = df.columns[0]
            y = df[first_col]
            X = df.drop(first_col, axis=1)

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed)

        print(f"Using {n_folds}-fold cross-validation with stratification")
        return separator, X, y, skf
    except Exception as e:
        print(f"Error preparing cross-validation: {e}")
        return None, None, None, None


def generate_higher_order_features(train_path, test_path, base_name_without_ext, fold_idx, max_z, minf, fold_prefix=""):
    """
    Generate higher-order features for train and test sets.
    Returns dictionaries mapping z values to file paths.
    """
    train_paths = {1: train_path}
    test_paths = {1: test_path}

    for z in range(2, max_z+1):
        train_z_out_path = os.path.join(
            os.path.dirname(train_path), f"z_{z}_{fold_prefix}{base_name_without_ext}_train.csv")
        cmd = f"python3 tabularbinary_to_tabularbinary_z.py -db {train_path} -od {train_z_out_path} -z {z} -minf {minf} -v 0"
        execute_cmd(cmd)
        train_paths[z] = train_z_out_path

        test_z_out_path = os.path.join(
            os.path.dirname(test_path), f"z_{z}_{fold_prefix}{base_name_without_ext}_test.csv")
        cmd = f"python3 tabularbinary_to_tabularbinary_z.py -db {test_path} -od {test_z_out_path} -z {z} -minf {minf} -v 0"
        execute_cmd(cmd)
        test_paths[z] = test_z_out_path

    return train_paths, test_paths


def run_experiments(train_paths, fold_results_file, fold_prefix, base_name_without_ext, fold_idx,
                    dataset, exp_parameters, max_k, max_z, num_workers, num_runs, seed, use_cv,
                    constant_m=0, verbose=0):
    """
    Run all experiments for a fold using multiprocessing.
    Returns list of async results.
    """
    pool = Pool(processes=num_workers)
    parallel_res = []

    fold_db_pref = os.path.join(
        os.path.dirname(train_paths[1]), f"{fold_prefix}{base_name_without_ext}_par_exact_")

    cmd = f"python3 main.py -db {train_paths[max_z]} -k {max_k} -op {fold_db_pref} -v 0 -ores {fold_results_file} -exact 1"
    if seed is not None:
        cmd += f" -seed {seed}"
    res = pool.apply_async(execute_cmd, (cmd,))
    parallel_res.append(res)

    j = 0
    for run_id in range(num_runs):
        print(f"{'Fold '+str(fold_idx)+' - ' if use_cv else ''}Dataset: {dataset}")
        for params in exp_parameters:
            theta = params[0]
            eps = params[1]

            if constant_m == 1:
                k_values = [max_k]
                z_values = [max_z]
                if verbose > 1:
                    print(
                        f"Using constant M: Only running with k={max_k}, z={max_z}")
            else:
                k_values = range(1, max_k+1)
                z_values = range(1, max_z+1)

            for k in k_values:
                for z in z_values:
                    print(
                        f"NEW EXPERIMENT: {'Fold '+str(fold_idx)+' - ' if use_cv else ''}{dataset} t={theta} k={k} z={z} e={eps}")

                    db_path = train_paths[z]

                    fold_db_pref = os.path.join(
                        os.path.dirname(train_paths[1]), f"{fold_prefix}{base_name_without_ext}_par_{j}_")

                    j += 1
                    if seed is not None:
                        derived_seed = seed + j + \
                            (fold_idx * 1000 if use_cv else 0)
                        cmd = f"python3 main.py -db {db_path} -theta {theta} -k {k} -epsilon {eps} -op {fold_db_pref} -v 0 -ores {fold_results_file} -seed {derived_seed}"
                    else:
                        cmd = f"python3 main.py -db {db_path} -theta {theta} -k {k} -epsilon {eps} -op {fold_db_pref} -v 0 -ores {fold_results_file}"
                    res = pool.apply_async(execute_cmd, (cmd,))
                    parallel_res.append(res)

    return pool, parallel_res


def calculate_performance(fold_results_file, fold_results_dir, train_path, num_workers, fold_idx=None, use_cv=False):
    """
    Calculate performance metrics and generate visualizations for a fold.
    Returns the path to the performance file.
    """
    fold_performance_file = os.path.join(
        fold_results_dir, "results_performance.csv")
    fold_dir_path = os.path.dirname(train_path)
    performance_cmd = f"python3 process_output_csv.py -fn {fold_results_file} -o {fold_performance_file} -j {num_workers} -exp_dir {fold_dir_path} -v 1"

    execute_cmd(performance_cmd)
    return fold_performance_file


def generate_visualization(fold_performance_file, train_path, fold_idx=None, use_cv=False):
    """
    Generate visualization dashboard for a fold.
    Returns the path to the visualization file.
    """
    fold_viz_path = os.path.join(os.path.dirname(train_path), "dashboard.html")
    viz_cmd = f"python3 visualization_bokeh.py -fn {fold_performance_file} -o {fold_viz_path} -v 1"
    execute_cmd(viz_cmd)

    print(f"- {'Fold '+str(fold_idx)+' visualization' if use_cv and fold_idx is not None else 'Visualization'} saved to: {fold_viz_path}\n")
    return fold_viz_path


def combine_fold_results(fold_results, exp_dir):
    """
    Combine results from multiple folds into a single dashboard.
    Uses the specific list of performance files from fold_results.
    Returns paths to combined files.
    """
    combined_perf_file = os.path.join(
        exp_dir, "results_performance_all_folds.csv")

    all_fold_data = []
    performance_files = []

    for fold_info in fold_results:
        fold_df = pd.read_csv(fold_info['performance_file'], sep=';')
        fold_df['fold'] = f"fold_{fold_info['fold']}"
        all_fold_data.append(fold_df)
        performance_files.append(fold_info['performance_file'])

    if all_fold_data:
        combined_df = pd.concat(all_fold_data, ignore_index=True)
        combined_df.to_csv(combined_perf_file, index=False, sep=';')

        combined_viz_path = os.path.join(exp_dir, "combined_dashboard.html")

        files_arg = ','.join(performance_files)
        combined_viz_cmd = f"python3 multi_experiment_dashboard.py -files {files_arg} -o {combined_viz_path} -v 1"
        execute_cmd(combined_viz_cmd)

        print(f"- Combined performance data saved to: {combined_perf_file}")
        print(
            f"- Combined visualization dashboard saved to: {combined_viz_path}")

        return combined_perf_file, combined_viz_path
    else:
        print("Warning: No fold data to combine")
        return None, None


def process_fold(fold_idx, dataset, exp_dir, base_name_without_ext, train_path, test_path,
                 max_z, minf, fold_results_file, exp_parameters, max_k, num_runs,
                 num_workers, seed, use_cv, constant_m=0, verbose=0):
    """
    Process a single fold (or the entire dataset if not using cross-validation).
    Returns dictionary with fold results.
    """
    fold_prefix = f"fold_{fold_idx}_" if use_cv else ""

    print(
        f"\n{'='*80}\nProcessing {'Fold '+str(fold_idx) if use_cv else 'Dataset'}\n{'='*80}")

    train_paths, test_paths = generate_higher_order_features(
        train_path, test_path, base_name_without_ext, fold_idx, max_z, minf, fold_prefix)

    pool, parallel_res = run_experiments(
        train_paths, fold_results_file, fold_prefix, base_name_without_ext,
        fold_idx, dataset, exp_parameters, max_k, max_z, num_workers,
        num_runs, seed, use_cv, constant_m, verbose)

    for res in tqdm(parallel_res):
        res.get()

    pool.close()
    pool.join()

    fold_results_dir = os.path.dirname(fold_results_file)
    print(
        f"\nCalculating performance metrics on {'fold '+str(fold_idx)+' ' if use_cv else ''}test set...")
    fold_performance_file = calculate_performance(
        fold_results_file, fold_results_dir, train_path, num_workers, fold_idx, use_cv)

    print(
        f"\nGenerating visualization for {'fold '+str(fold_idx) if use_cv else 'experiment'}...")
    fold_viz_path = generate_visualization(
        fold_performance_file, train_path, fold_idx, use_cv)

    return {
        'fold': fold_idx,
        'results_file': fold_results_file,
        'performance_file': fold_performance_file,
        'visualization_path': fold_viz_path
    }
