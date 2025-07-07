import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


output_dir = "results_amount_of_rule_lists_per_k"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")


base_dir = "results_experiment3"


k_value_rule_counts = defaultdict(
    lambda: defaultdict(lambda: defaultdict(set)))
total_rule_lists = defaultdict(lambda: defaultdict(int))


for dataset in os.listdir(base_dir):
    dataset_path = os.path.join(base_dir, dataset)
    if not os.path.isdir(dataset_path):
        continue

    print(f"Processing dataset: {dataset}")

    for fold in range(5):
        fold_dir = os.path.join(dataset_path, f"fold_{fold}", "results")
        results_file = os.path.join(fold_dir, "results_performance.csv")

        if not os.path.exists(results_file):
            print(f"File not found: {results_file}")
            continue

        print(f"  Processing fold_{fold}")

        try:

            df = pd.read_csv(results_file, sep=';')

            fold_rules = set()

            for _, row in df.iterrows():
                rule = row['rule']
                fold_rules.add(rule)

                try:
                    k_values = ast.literal_eval(row['k_values'])

                    for k in k_values:
                        k_value_rule_counts[dataset][fold][k].add(rule)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing k_values: {row['k_values']} - {e}")

            total_rule_lists[dataset][fold] = len(fold_rules)

        except Exception as e:
            print(f"Error processing {results_file}: {e}")


data_for_plot = []
for dataset, fold_dict in k_value_rule_counts.items():
    for fold, k_dict in fold_dict.items():
        for k, rule_set in k_dict.items():
            data_for_plot.append({
                'dataset': dataset,
                'fold': fold,
                'k_value': k,
                'rule_list_count': len(rule_set)
            })


plot_df = pd.DataFrame(data_for_plot)


for dataset in plot_df['dataset'].unique():
    dataset_df = plot_df[plot_df['dataset'] == dataset]

    plt.figure(figsize=(15, 8))

    pivot_df = dataset_df.pivot_table(
        index='k_value', columns='fold', values='rule_list_count', fill_value=0)
    pivot_df.sort_index(inplace=True)

    ax = pivot_df.plot(kind='bar', figsize=(15, 8))
    plt.title(f"{dataset}: Number of Unique Rule Lists by k-value for Each Fold")
    plt.xlabel('k value')
    plt.ylabel('Count of Unique Rule Lists')
    plt.legend(title='Fold', labels=[f'Fold {i + 1}' for i in range(5)])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_file = os.path.join(
        output_dir, f"{dataset}_rule_lists_by_k_value.pdf")
    plt.savefig(output_file, format='pdf')
    print(f"Saved: {output_file}")
    plt.close()


plt.figure(figsize=(15, 10))
for i, dataset in enumerate(sorted(plot_df['dataset'].unique())):
    plt.subplot(len(plot_df['dataset'].unique()), 1, i+1)

    agg_df = plot_df[plot_df['dataset'] == dataset].groupby(
        'k_value')['rule_list_count'].sum().reset_index()
    agg_df = agg_df.sort_values('k_value')

    sns.barplot(data=agg_df, x='k_value', y='rule_list_count')
    plt.title(
        f"{dataset}: Total Number of Unique Rule Lists by k-value (Across All Folds)")
    plt.xlabel('k value')
    plt.ylabel('Count of Unique Rule Lists')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()


combined_output_file = os.path.join(
    output_dir, "all_datasets_rule_lists_by_k_value.pdf")
plt.savefig(combined_output_file, format='pdf')
print(f"Saved: {combined_output_file}")
plt.close()


csv_output_file = os.path.join(
    output_dir, "rule_lists_by_k_value_detailed.csv")
plot_df.to_csv(csv_output_file, index=False)
print(f"Saved: {csv_output_file}")


plt.figure(figsize=(15, 8))


avg_data = []
for dataset in plot_df['dataset'].unique():
    dataset_df = plot_df[plot_df['dataset'] == dataset]
    dataset_avg = dataset_df.groupby(
        'k_value')['rule_list_count'].mean().reset_index()
    for k in dataset_avg['k_value']:

        if k == 13:
            dataset_avg.loc[dataset_avg['k_value']
                            == k, 'rule_list_count'] -= 1
    dataset_avg['dataset'] = dataset
    avg_data.append(dataset_avg)


avg_df = pd.concat(avg_data)


plt.figure(figsize=(15, 8))
sns.lineplot(data=avg_df, x='k_value', y='rule_list_count',
             hue='dataset', marker='o', linewidth=2.5)

plt.title('Average Number of Unique Rule Lists by k-value per fold: Comparison Across Datasets', fontsize=16)
plt.xlabel('k value', fontsize=14)
plt.ylabel('Average Count of Unique Rule Lists per fold', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Dataset', fontsize=12, title_fontsize=14)


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.tight_layout()


comparison_output_file = os.path.join(
    output_dir, "dataset_comparison_avg_rule_lists.pdf")
plt.savefig(comparison_output_file, format='pdf')
print(f"Saved: {comparison_output_file}")
plt.close()

print("Analysis complete. Visualizations and CSV summary have been saved to the directory:", output_dir)
