import pandas as pd
import numpy as np
import argparse
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import os


def select_features(
    input_file,
    output_file,
    num_features=25,
    target_column="{T}",
    method='random_forest',
    sample_size=None,
    plot=False,
    sampling_strategy='random'
):
    print(f"Loading data from {input_file}...")

    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    df_peek = pd.read_csv(input_file, nrows=5)

    if target_column not in df_peek.columns:
        print(
            f"Warning: Target column '{target_column}' not found. Available columns: {df_peek.columns.tolist()}")
        if "{T}" in df_peek.columns:
            target_column = "{T}"
            print(f"Using '{target_column}' as target column")
        else:
            target_column = df_peek.columns[0]
            print(f"Using first column '{target_column}' as target")

    print(f"Using target column: '{target_column}'")

    if sample_size:
        if sampling_strategy == 'random':
            print(
                f"Using memory-efficient random sampling (nrows={sample_size})")
            df = pd.read_csv(input_file, nrows=sample_size)
            print(f"Dataset shape: {df.shape} (rows, columns)")

        elif sampling_strategy == 'balanced':
            print("Using chunked balanced sampling to save memory...")

            class_counts = {}
            chunk_size = min(100000, sample_size * 5)
            for chunk in pd.read_csv(input_file, chunksize=chunk_size):
                counts = chunk[target_column].value_counts().to_dict()
                for k, v in counts.items():
                    class_counts[k] = class_counts.get(k, 0) + v
                del chunk
                gc.collect()

            print(f"Class distribution in dataset: {class_counts}")

            classes = list(class_counts.keys())
            samples_per_class = {cls: min(sample_size // len(classes), class_counts[cls])
                                 for cls in classes}

            remainder = sample_size - sum(samples_per_class.values())
            for cls in classes:
                if remainder <= 0:
                    break
                additional = min(
                    remainder, class_counts[cls] - samples_per_class[cls])
                samples_per_class[cls] += additional
                remainder -= additional

            print(f"Samples per class: {samples_per_class}")

            sampled_dfs = []
            for cls in classes:
                if samples_per_class[cls] <= 0:
                    continue

                cls_needed = samples_per_class[cls]
                cls_collected = 0

                for chunk in pd.read_csv(input_file, chunksize=chunk_size):
                    if cls_needed <= 0:
                        break

                    cls_chunk = chunk[chunk[target_column] == cls]
                    take_n = min(cls_needed, len(cls_chunk))

                    if take_n > 0:
                        sampled_dfs.append(
                            cls_chunk.sample(take_n, random_state=42))
                        cls_needed -= take_n
                        cls_collected += take_n

                    del chunk
                    del cls_chunk
                    gc.collect()

                print(f"Collected {cls_collected} samples for class {cls}")

            df = pd.concat(sampled_dfs)

            del sampled_dfs
            gc.collect()

            print(
                f"Final class distribution: {df[target_column].value_counts().to_dict()}")

        elif sampling_strategy == 'informative':

            print("Warning: Informative sampling requires more memory.")
            print("Using a smaller subset to build the model...")

            subsample = min(50000, sample_size * 3)
            df_subset = pd.read_csv(input_file, nrows=subsample)

            X_subset = df_subset.drop(columns=[target_column])
            y_subset = df_subset[target_column]

            if not pd.api.types.is_numeric_dtype(y_subset):
                y_subset = pd.factorize(y_subset)[0]

            categorical_columns = X_subset.select_dtypes(
                include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                X_subset_encoded = pd.get_dummies(X_subset, drop_first=True)
            else:
                X_subset_encoded = X_subset

            print("Training quick model on subset...")
            quick_model = RandomForestClassifier(
                n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
            quick_model.fit(X_subset_encoded, y_subset)

            del X_subset_encoded
            gc.collect()

            print("Scoring entire dataset in chunks to find informative samples...")
            uncertainty_scores = []
            row_indices = []

            chunk_size = 10000
            row_idx = 0

            for chunk in pd.read_csv(input_file, chunksize=chunk_size):

                X_chunk = chunk.drop(columns=[target_column])

                if len(categorical_columns) > 0:
                    X_chunk_encoded = pd.get_dummies(X_chunk, drop_first=True)

                    missing_cols = set(X_subset_encoded.columns) - \
                        set(X_chunk_encoded.columns)
                    for col in missing_cols:
                        X_chunk_encoded[col] = 0
                    X_chunk_encoded = X_chunk_encoded[X_subset_encoded.columns]
                else:
                    X_chunk_encoded = X_chunk

                probs = quick_model.predict_proba(X_chunk_encoded)

                if probs.shape[1] == 2:
                    chunk_scores = 1 - np.abs(2 * probs[:, 1] - 1)
                else:
                    chunk_scores = np.sum(-probs *
                                          np.log2(probs + 1e-10), axis=1)

                uncertainty_scores.extend(chunk_scores)
                row_indices.extend(range(row_idx, row_idx + len(chunk)))
                row_idx += len(chunk)

                del X_chunk, X_chunk_encoded, chunk, probs, chunk_scores
                gc.collect()

            uncertainty_data = list(zip(row_indices, uncertainty_scores))
            uncertainty_data.sort(key=lambda x: x[1], reverse=True)

            uncertain_size = int(sample_size * 0.7)
            top_uncertain = [idx for idx,
                             _ in uncertainty_data[:uncertain_size]]

            remaining = list(set(row_indices) - set(top_uncertain))
            random_size = min(sample_size - uncertain_size, len(remaining))
            random_indices = np.random.choice(
                remaining, size=random_size, replace=False)

            selected_indices = np.concatenate([top_uncertain, random_indices])

            print(
                f"Loading {len(selected_indices)} selected informative rows...")
            df = pd.read_csv(input_file, skiprows=lambda i: i >
                             0 and i-1 not in selected_indices)
    else:

        if file_size_mb > 500:
            print(
                f"WARNING: Loading large file ({file_size_mb:.1f} MB) without sampling.")
            print("This may cause memory issues. Consider using -s option.")

        df = pd.read_csv(input_file)
        print(f"Dataset shape: {df.shape} (rows, columns)")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    categorical_columns = X.select_dtypes(
        include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        X = pd.get_dummies(X, drop_first=True)
    if method == 'random_forest':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.factorize(y)[0]
        model = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
    elif method == 'mutual_info':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.factorize(y)[0]
        importances = mutual_info_classif(X, y, random_state=42)
    elif method == 'correlation':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.factorize(y)[0]
        importances = np.zeros(X.shape[1])
        for i, col in enumerate(X.columns):
            importances[i] = abs(np.corrcoef(X[col], y)[0, 1])
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    top_features = feature_importance.head(num_features)['feature'].tolist()
    original_features = set()
    for feature in top_features:
        if '' in feature and any(f"{col}" in feature for col in categorical_columns):
            for col in categorical_columns:
                if feature.startswith(f"{col}_"):
                    original_feature = col
                    break
        else:
            original_feature = feature
        original_features.add(original_feature)
    original_features = list(original_features)
    if len(original_features) > num_features:
        original_features = original_features[:num_features]
    columns_to_keep = [target_column] + original_features
    reduced_df = df[columns_to_keep]
    print(
        f"\nSelected {len(original_features)} features out of {X.shape[1]} encoded features")
    print(f"Features selected: {', '.join(original_features)}")
    print(f"\nReduced dataset shape: {reduced_df.shape}")
    reduced_df.to_csv(output_file, index=False)
    print(f"Saved reduced dataset to {output_file}")
    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(range(min(20, len(feature_importance))),
                feature_importance['importance'][:20])
        plt.xticks(range(min(20, len(feature_importance))),
                   feature_importance['feature'][:20], rotation=90)
        plt.title(f'Top 20 Feature Importance ({method})')
        plt.tight_layout()
        plot_file = output_file.replace('.csv', '_importance.png')
        plt.savefig(plot_file)
        print(f"Saved feature importance plot to {plot_file}")


print("Feature Selection Script")
parser = argparse.ArgumentParser(
    description='Select most important features from a dataset')
parser.add_argument('--input', '-i', required=True, help='Input CSV file')
parser.add_argument('--output', '-o', required=True,
                    help='Output CSV file')
parser.add_argument('--num_features', '-n', type=int,
                    default=25, help='Number of features to keep')
parser.add_argument(
    '--target', '-t', default="{T}", help='Target column name (default: {T})')
parser.add_argument('--method', '-m', choices=['random_forest', 'mutual_info', 'correlation'],
                    default='random_forest', help='Feature selection method')
parser.add_argument('--sample', '-s', type=int,
                    help='Number of rows to sample from dataset')
parser.add_argument('--sampling', choices=['random', 'balanced', 'informative'],
                    default='balanced', help='Sampling strategy to use')
parser.add_argument('--plot', '-p', action='store_true',
                    help='Generate feature importance plot')
args = parser.parse_args()
select_features(
    args.input,
    args.output,
    args.num_features,
    args.target,
    args.method,
    args.sample,
    args.plot,
    args.sampling
)
