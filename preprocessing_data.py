import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

print("\n=== LOADING DATA ===")
df = pd.read_csv("raw_data/creditcard.csv")
print(f"Original dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")

X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeature statistics sample (first 5 features):")
print(X.iloc[:, :5].describe().round(2))


def supervised_binarize(X, y, max_depth=1):
    print(f"\n=== BINARIZING FEATURES (max_depth={max_depth}) ===")
    binary_features = pd.DataFrame()
    feature_names = []
    bins_per_feature = {}

    for col in X.columns:
        print(f"\nProcessing feature: {col}")

        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=1000)
        dt.fit(X[[col]], y)

        tree = dt.tree_

        print(f"  Tree node count: {tree.node_count}")

        thresholds_for_feature = []
        if tree.node_count > 1:
            for node_id in range(tree.node_count):
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    threshold = tree.threshold[node_id]
                    thresholds_for_feature.append(threshold)

                    binary_col = (X[col] > threshold).astype(int)
                    feature_name = f"{col}_>{threshold:.4f}"
                    binary_features[feature_name] = binary_col
                    feature_names.append(feature_name)
                    print(
                        f"  Added binary feature: {feature_name} (1s: {binary_col.sum()}, 0s: {len(binary_col) - binary_col.sum()})")

            bins_per_feature[col] = len(thresholds_for_feature)
        else:
            print(f"  Skipped {col} (tree made no split)")
            bins_per_feature[col] = 0

    print(f"\nTotal binary features created: {len(feature_names)}")
    print(
        f"Average bins per feature: {np.mean(list(bins_per_feature.values())):.2f}")
    print(
        f"Features with most bins: {sorted(bins_per_feature.items(), key=lambda x: x[1], reverse=True)[:3]}")

    return binary_features, feature_names


print("\n=== STARTING BINARIZATION ===")
X_bin, bin_feature_names = supervised_binarize(X, y, max_depth=2)
print(f"\nBinarized data shape: {X_bin.shape}")
print("\nSample of binarized data (5 rows, 5 columns):")
print(X_bin.iloc[:5, :5])


print("\n=== FEATURE SELECTION ===")
k = 50
print(f"Selecting top {k} features using mutual information")
selector = SelectKBest(mutual_info_classif, k=k)
X_selected = selector.fit_transform(X_bin, y)


selected_features = X_bin.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]


print("\nTop 10 selected features by mutual information:")
for feature, score in sorted(zip(selected_features, feature_scores), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feature}: {score:.6f}")

X_bin_final = pd.DataFrame(X_selected, columns=selected_features)
print(f"\nFinal selected data shape: {X_bin_final.shape}")


df_final = pd.concat([y.rename("{T}"), X_bin_final], axis=1)

print("\n=== FINAL DATASET ===")
print(f"Final dataset shape: {df_final.shape}")
print("Sample rows (5 rows, first few columns):")
print(df_final.iloc[:5, :min(6, df_final.shape[1])])

print("\nClass distribution in final dataset:")
print(df_final["{T}"].value_counts())


output_file = "corels_input_combined.csv"
df_final.to_csv(output_file, index=False)
print(f"\nSaved preprocessed data to: {output_file}")
