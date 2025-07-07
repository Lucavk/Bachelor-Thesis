import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from time import time
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import re


parser = argparse.ArgumentParser()
parser.add_argument("-fn", help="FileName (csv file), e.g. 'results.csv'",
                    default="results.csv")
parser.add_argument("-o", help="Output file name",
                    default="results_performance.csv")
parser.add_argument("-v", help="Verbose level (0-2)", type=int, default=1)
parser.add_argument(
    "-j", help="Number of parallel jobs (default: number of CPU cores)", type=int, default=None)
parser.add_argument("-exp_dir", help="Experiment directory path",
                    default=None)
parser.add_argument("-keep_dupl", type=int, default=0,
                    help="Keep duplicate rules (1) or merge them (0) (default: 0)")
args = parser.parse_args()


n_jobs = args.j if args.j is not None else cpu_count()
if args.v > 0:
    print(f"Using {n_jobs} parallel jobs")


results_csv = pd.read_csv(args.fn, sep=";")
if args.v > 0:
    print(f"Loaded {len(results_csv)} results from {args.fn}")


required_columns = ['dataset', 'k', 'opt_rule']
for col in required_columns:
    if col not in results_csv.columns:
        print(f"Error: Required column '{col}' not found in results CSV")
        exit(1)


def get_test_file_path(train_path):
    """Convert train path to corresponding test path"""
    return train_path.replace("_train.csv", "_test.csv")


def extract_z_value(dataset_path):
    """Extract z value from dataset path"""

    match = re.search(r'/z_(\d+)_', dataset_path)
    if match:
        return int(match.group(1))
    else:

        return 1


def parse_rule_list(rule_list):
    """Parse a rule list string into a list of condition-prediction pairs"""
    rules_str = rule_list.split(", ")
    parsed_rules = []

    for i in range(len(rules_str)):
        rule_str = rules_str[i]
        if "then" in rule_str:
            items = rule_str.split(" then ")
            cond = items[0]
            pred = items[1]
            cond = cond.replace("else if (", "")
            cond = cond.replace("if (", "")
            cond = cond.replace(")", "")
            pred = pred.replace("(", "")
            pred = pred.replace(")", "")
        else:
            cond = ""
            pred = rule_str.replace("else (", "")
            pred = pred.replace(")", "")
        parsed_rules.append((cond, pred))

    if args.v > 1:
        print(f"Parsed rules: {parsed_rules}")

    return parsed_rules


def apply_rule_list(df, cond_pred_list):
    """Apply a rule list to a dataframe and return predictions"""
    predictions = np.zeros(len(df))
    not_covered = df.copy()

    for (cond, pred) in cond_pred_list:
        if pred == "{T=1}":
            pred_val = 1
        else:
            pred_val = 0

        if len(cond) > 0:
            try:

                if cond in not_covered.columns:
                    covered = not_covered[cond] == 1

                else:
                    try:
                        feature_idx = int(cond)
                        column_names = list(not_covered.columns)

                        if "{T}" in column_names:
                            column_names.remove("{T}")
                        if feature_idx < len(column_names):
                            column_name = column_names[feature_idx]
                            covered = not_covered[column_name] == 1
                        else:
                            print(
                                f"Warning: Feature index {feature_idx} out of bounds")
                            covered = pd.Series(False, index=not_covered.index)
                    except ValueError:

                        if "and" in cond:
                            subconds = cond.split(" and ")
                            covered = pd.Series(True, index=not_covered.index)
                            for subcond in subconds:
                                subcond = subcond.strip()
                                if subcond in not_covered.columns:
                                    covered &= (not_covered[subcond] == 1)
                                else:
                                    try:
                                        feature_idx = int(subcond)
                                        column_name = list(not_covered.columns)[
                                            feature_idx]
                                        covered &= (
                                            not_covered[column_name] == 1)
                                    except (ValueError, IndexError):
                                        print(
                                            f"Warning: Could not parse condition '{subcond}'")
                                        covered &= False
                        else:
                            print(
                                f"Warning: Could not parse condition '{cond}'")
                            covered = pd.Series(False, index=not_covered.index)
            except Exception as e:
                print(f"Error applying rule condition '{cond}': {str(e)}")
                covered = pd.Series(False, index=not_covered.index)

            covered_idx = not_covered.index[covered]
            not_covered = not_covered.loc[~covered]
        else:
            covered_idx = not_covered.index

        predictions[covered_idx] = pred_val

    return predictions


def evaluate_metrics(y_true, y_pred, rule_str):
    """Calculate various performance metrics"""
    metrics = {}

    metrics['rule'] = rule_str

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    try:
        metrics['precision'] = precision_score(y_true, y_pred)
    except Exception as e:
        metrics['precision'] = np.nan

    try:
        metrics['recall'] = recall_score(y_true, y_pred)
    except Exception as e:
        metrics['recall'] = np.nan

    try:
        metrics['f1_score'] = f1_score(y_true, y_pred)
    except Exception as e:
        metrics['f1_score'] = np.nan

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    except Exception as e:
        metrics['roc_auc'] = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception as e:
        metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = metrics['specificity'] = np.nan

    parsed_rules = parse_rule_list(rule_str)
    metrics['num_rules'] = len(parsed_rules)
    metrics['loss_db'] = 1 - metrics['accuracy'] + \
        0.0001 * (len(parsed_rules) - 1)

    return metrics


enriched_results = []
for idx, row in results_csv.iterrows():
    train_path = row['dataset']
    z_value = extract_z_value(train_path)
    test_path = get_test_file_path(train_path)

    enriched_row = row.copy()
    enriched_row['true_z'] = z_value
    enriched_row['test_path'] = test_path

    enriched_row['is_exact'] = False
    if 'm' in row and 'exact' in row:
        if row['m'] == 0 or row['exact'] == 1:
            enriched_row['is_exact'] = True

    enriched_results.append(enriched_row)


enriched_df = pd.DataFrame(enriched_results)

all_metrics = []
start_time = time()


if args.keep_dupl == 1:

    if args.v > 0:
        print("Processing all rules individually (keeping duplicates)")

    for idx, row in tqdm(enriched_df.iterrows(), total=len(enriched_df), desc="Processing rules"):
        try:
            rule = row['opt_rule']
            is_exact = row['is_exact']
            train_path = row['dataset']
            test_path = row['test_path']
            k_value = row['k']
            z_value = row['true_z']

            if not os.path.exists(test_path):
                print(f"Warning: Test file not found at {test_path}")
                continue

            try:
                test_df = pd.read_csv(test_path)
                test_df = test_df.astype(int)
            except Exception as e:
                print(f"Error loading test dataset {test_path}: {str(e)}")
                continue

            if "{T}" not in test_df.columns:
                print(f"Error: Target column '{{T}}' not found in {test_path}")
                continue

            parsed_rule = parse_rule_list(rule)
            y_true = test_df["{T}"]

            y_pred = apply_rule_list(test_df, parsed_rule)
            metrics = evaluate_metrics(y_true, y_pred, rule)

            metrics['dataset'] = train_path
            metrics['test_dataset'] = test_path
            metrics['avg_k'] = k_value
            metrics['avg_z'] = z_value
            metrics['k_values'] = str([k_value])
            metrics['z_values'] = str([z_value])
            metrics['occurrence_count'] = 1
            metrics['is_exact'] = is_exact

            all_metrics.append(metrics)

        except Exception as e:
            print(f"Error processing rule at index {idx}: {str(e)}")
            continue
else:

    if args.v > 0:
        print("Grouping duplicate rules (default behavior)")

    rule_groups = enriched_df.groupby(['opt_rule', 'is_exact'])

    for (rule, is_exact), group in tqdm(rule_groups, desc="Processing unique rules"):
        try:

            first_row = group.iloc[0]
            train_path = first_row['dataset']
            test_path = first_row['test_path']

            avg_k = group['k'].mean()
            avg_z = group['true_z'].mean()

            k_values = sorted(group['k'].unique())
            z_values = sorted(group['true_z'].unique())

            if not os.path.exists(test_path):
                print(f"Warning: Test file not found at {test_path}")
                continue

            try:
                test_df = pd.read_csv(test_path)
                test_df = test_df.astype(int)
            except Exception as e:
                print(f"Error loading test dataset {test_path}: {str(e)}")
                continue

            if "{T}" not in test_df.columns:
                print(f"Error: Target column '{{T}}' not found in {test_path}")
                continue

            parsed_rule = parse_rule_list(rule)
            y_true = test_df["{T}"]

            y_pred = apply_rule_list(test_df, parsed_rule)
            metrics = evaluate_metrics(y_true, y_pred, rule)

            metrics['dataset'] = train_path
            metrics['test_dataset'] = test_path
            metrics['avg_k'] = avg_k
            metrics['avg_z'] = avg_z
            metrics['k_values'] = str(k_values)
            metrics['z_values'] = str(z_values)
            metrics['occurrence_count'] = len(group)
            metrics['is_exact'] = is_exact

            all_metrics.append(metrics)

        except Exception as e:
            print(
                f"Error processing rule '{rule}' (exact={is_exact}): {str(e)}")
            continue


if not all_metrics:
    print("Warning: No metrics were successfully calculated.")
    metrics_df = pd.DataFrame(columns=['dataset', 'test_dataset', 'rule', 'avg_k', 'avg_z',
                                       'accuracy', 'precision', 'recall', 'f1_score',
                                       'roc_auc', 'specificity', 'loss_db', 'is_exact'])
else:
    metrics_df = pd.DataFrame(all_metrics)

runtime = time() - start_time
if args.v > 0:
    print(f"Total processing time: {runtime:.2f} seconds")
    print(f"Processed {len(metrics_df)} {'individual' if args.keep_dupl == 1 else 'unique'} rules from {len(results_csv)} total rules")
    if args.keep_dupl != 1:
        print(
            f"Speedup factor: {len(results_csv)/max(1, len(metrics_df)):.2f}x")

    exact_count = metrics_df['is_exact'].sum()
    approximate_count = len(metrics_df) - exact_count
    print(
        f"Found {exact_count} exact solution(s) and {approximate_count} approximate solution(s)")


metrics_df.to_csv(args.o, index=False, sep=';')
if args.v > 0:
    print(f"Results saved to {args.o}")


if not metrics_df.empty and args.v > 0:
    try:

        metrics_df['k_group'] = metrics_df['avg_k'].round().astype(int)
        metrics_df['z_group'] = metrics_df['avg_z'].round().astype(int)

        for solution_type, label in [(True, "EXACT"), (False, "APPROXIMATE")]:
            subset = metrics_df[metrics_df['is_exact'] == solution_type]
            if len(subset) > 0:
                summary = subset.groupby(['k_group', 'z_group']).agg({
                    'accuracy': ['mean', 'std'],
                    'precision': ['mean', 'std'],
                    'recall': ['mean', 'std'],
                    'f1_score': ['mean', 'std'],
                    'loss_db': ['mean', 'std'],
                    'occurrence_count': 'sum'
                })
                print(
                    f"\nPerformance summary for {label} solutions by rule complexity (k) and feature complexity (z):")
                print(summary)
    except Exception as e:
        print(f"Error creating summary statistics: {str(e)}")
