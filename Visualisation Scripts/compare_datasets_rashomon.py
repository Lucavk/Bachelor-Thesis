import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from bokeh.plotting import figure, output_file, save
from bokeh.models import (ColumnDataSource, HoverTool,
                          Legend, LegendItem, Div)
from bokeh.layouts import column, row
from bokeh.palettes import Category20, Turbo256
from bokeh.io import export_svg
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

parser = argparse.ArgumentParser(
    description="Compare Rashomon sets across different datasets")
parser.add_argument("-d", "--directory", default="results_experiment2",
                    help="Base directory containing experiment results (default: results_experiments)")
parser.add_argument("-o", "--output", default="dataset_comparison.html",
                    help="Output HTML file (default: dataset_comparison.html)")
parser.add_argument("-f", "--fold", type=int, default=0,
                    help="Fold number to use for comparison (default: 0)")
parser.add_argument("-constant", type=int, default=0,
                    help="Process folders ending with 'Constant' (1) or not (0) (default: 0)")
parser.add_argument("-v", "--verbose", type=int, default=1,
                    help="Verbose level (0-2) (default: 1)")
args = parser.parse_args()


def find_experiment_dirs(base_dir, constant_suffix=False):
    """
    Find all experiment directories that match the constant suffix criteria.
    """
    experiment_dirs = []

    if not os.path.isdir(base_dir):
        print(f"Base directory {base_dir} does not exist")
        return []

    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):
            is_constant = item.endswith("Constant")
            if constant_suffix == is_constant:
                if args.fold == -1:

                    experiment_dirs.append(full_path)
                else:

                    fold_path = os.path.join(full_path, f"fold_{args.fold}")
                    results_file = os.path.join(
                        fold_path, "results", "results_performance.csv")
                    if os.path.isdir(fold_path) and os.path.exists(results_file):
                        experiment_dirs.append(full_path)

    return experiment_dirs


def load_dataset_fold_data(experiment_dir, fold_num):
    """
    Load data from a specific fold of a dataset.
    """
    dataset_name = os.path.basename(experiment_dir)
    results_file = os.path.join(
        experiment_dir, f"fold_{fold_num}", "results", "results_performance.csv")

    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file, sep=';')
            df['dataset_name'] = dataset_name

            if args.verbose > 1:
                print(
                    f"  Loaded {len(df)} rule lists from {dataset_name} fold_{fold_num}")

            return df
        except Exception as e:
            if args.verbose > 0:
                print(f"  Error loading {results_file}: {str(e)}")
    else:
        if args.verbose > 0:
            print(f"  Results file not found: {results_file}")

    return None


def add_derived_metrics(df):
    """
    Add derived metrics to the dataframe.
    """
    df['rule_length'] = df['rule'].apply(lambda x: len(x.split(', ')) - 1)

    if 'is_exact' in df.columns:
        df['is_exact'] = df['is_exact'].astype(bool)
    else:
        df['is_exact'] = False

    if 'fp' in df.columns and 'tn' in df.columns:
        df['false_positive_rate'] = df['fp'] / (df['fp'] + df['tn'])
        df['true_positive_rate'] = df['recall']

    return df


def create_normalized_dataset_comparison_plot(measure, title, datasets_data, show_all_folds=False):
    """
    Create normalized Rashomon set size comparison plot across datasets.
    """
    p = figure(
        title=f"Dataset: Normalized Rashomon Sets for {title}",
        height=500, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="threshold θ",
        y_axis_label="size of Rashomon set (percentage of all rules)",
        y_range=(0, 100)
    )

    num_datasets = len(datasets_data)
    if num_datasets <= 10:
        colors = Category20[20][:num_datasets*2:2]
    else:
        step = len(Turbo256) // num_datasets
        colors = [Turbo256[i*step] for i in range(num_datasets)]

    legend_items = []

    for i, (dataset_name, df) in enumerate(datasets_data.items()):
        dataset_name = dataset_name.replace("_", "").replace(
            "Constant", "").replace("reduced", "")
        if df is None or df.empty:
            continue

        display_name = dataset_name
        if display_name.endswith("Constant"):
            display_name = display_name[:-8]

        best_performance = df[measure].max()

        theta_values = np.linspace(0, 0.2, 100)

        if show_all_folds:

            min_percentages = []
            max_percentages = []
            for theta in theta_values:
                threshold = best_performance - theta
                fold_percentages = []
                for fold in df['fold'].unique():
                    fold_df = df[df['fold'] == fold]

                    fold_total = len(fold_df)
                    rashomon_size = len(fold_df[fold_df[measure] >= threshold])
                    percentage = (rashomon_size / fold_total *
                                  100) if fold_total > 0 else 0
                    fold_percentages.append(percentage)
                min_percentages.append(min(fold_percentages))
                max_percentages.append(max(fold_percentages))

            source = ColumnDataSource(data=dict(
                theta=np.concatenate([theta_values, theta_values[::-1]]),
                size=np.concatenate([max_percentages, min_percentages[::-1]]),
                dataset=[display_name] * (2 * len(theta_values))
            ))

            patch = p.patch('theta', 'size', source=source,
                            alpha=0.3, color=colors[i])
            legend_items.append(LegendItem(
                label=f"{display_name} (all folds)", renderers=[patch]))
        else:

            total_rules = len(df)
            norm_sizes = []
            abs_sizes = []
            threshold_values = []

            for theta in theta_values:
                threshold = best_performance - theta
                threshold_values.append(threshold)
                rashomon_size = len(df[df[measure] >= threshold])
                abs_sizes.append(rashomon_size)

                norm_size = (rashomon_size / total_rules *
                             100) if total_rules > 0 else 0
                norm_sizes.append(norm_size)

            source = ColumnDataSource(data=dict(
                theta=theta_values,
                size=norm_sizes,
                abs_size=abs_sizes,
                threshold=threshold_values,
                dataset=[display_name] * len(theta_values),
                total_rules=[total_rules] * len(theta_values),
                best_performance=[best_performance] * len(theta_values)
            ))

            line = p.line('theta', 'size', source=source,
                          line_width=2.5, line_color=colors[i], alpha=0.8)

            p.scatter('theta', 'size', source=source,
                      size=4, fill_color="white", line_color=colors[i], alpha=0.6)

            legend_items.append(LegendItem(
                label=f"{display_name} ({total_rules} rule lists)", renderers=[line]))

    hover = HoverTool(tooltips=[
        ("Dataset", "@dataset"),
        ("Theta", "@theta{0.000}"),
        ("% of Rules", "@size{0.0}%")
    ])
    p.add_tools(hover)

    legend = Legend(items=legend_items, location="center")
    p.add_layout(legend, 'right')

    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"
    return p


def load_all_folds_data(experiment_dir):
    """
    Load data from all folds of a dataset.
    """
    all_folds_data = []
    for fold_num in range(5):
        fold_data = load_dataset_fold_data(experiment_dir, fold_num)
        if fold_data is not None:
            fold_data['fold'] = fold_num
            all_folds_data.append(fold_data)
    if all_folds_data:
        return pd.concat(all_folds_data, ignore_index=True)
    return None


def create_dataset_comparison_plot(measure, title, datasets_data, show_all_folds=False):
    """
    Create Rashomon set size comparison plot across datasets.
    """
    p = figure(
        title=f"Dataset Comparison: Rashomon Set Sizes for {title}",
        height=500, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="θ (threshold)",
        y_axis_label="Rashomon Set Size |R(θ)|"
    )

    num_datasets = len(datasets_data)
    if num_datasets <= 10:
        colors = Category20[20][:num_datasets*2:2]
    else:
        step = len(Turbo256) // num_datasets
        colors = [Turbo256[i*step] for i in range(num_datasets)]

    legend_items = []

    for i, (dataset_name, df) in enumerate(datasets_data.items()):
        if df is None or df.empty:
            continue

        display_name = dataset_name
        if display_name.endswith("Constant"):
            display_name = display_name[:-8]

        total_rules = len(df)

        best_performance = df[measure].max()

        theta_values = np.linspace(0, 0.2, 100)

        if show_all_folds:

            min_sizes = []
            max_sizes = []
            for theta in theta_values:
                threshold = best_performance - theta
                fold_sizes = []
                for fold in df['fold'].unique():
                    fold_df = df[df['fold'] == fold]
                    fold_size = len(fold_df[fold_df[measure] >= threshold])
                    fold_sizes.append(fold_size)
                min_sizes.append(min(fold_sizes))
                max_sizes.append(max(fold_sizes))

            source = ColumnDataSource(data=dict(
                theta=np.concatenate([theta_values, theta_values[::-1]]),
                size=np.concatenate([max_sizes, min_sizes[::-1]]),
                dataset=[display_name] * (2 * len(theta_values))
            ))

            patch = p.patch('theta', 'size', source=source,
                            alpha=0.3, color=colors[i])
            legend_items.append(LegendItem(
                label=f"{display_name} (all folds)", renderers=[patch]))
        else:
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

            rashomon_sizes = []
            threshold_values = []

            for theta in theta_values:

                threshold = best_performance - theta
                threshold_values.append(threshold)

                rashomon_size = len(df[df[measure] >= threshold])
                rashomon_sizes.append(rashomon_size)

            source = ColumnDataSource(data=dict(
                theta=theta_values,
                size=rashomon_sizes,
                threshold=threshold_values,
                dataset=[display_name] * len(theta_values),
                total_rules=[total_rules] * len(theta_values),
                best_performance=[best_performance] * len(theta_values)
            ))

            line = p.line(
                'theta', 'size', source=source,
                line_width=2.5, line_color=colors[i],
                line_alpha=0.8
            )

            p.scatter(
                x='theta', y='size', source=source,
                size=4, fill_color="white", line_color=colors[i],
                alpha=0.6
            )

            legend_items.append(LegendItem(
                label=f"{display_name} ({total_rules} rules)", renderers=[line]))

    hover = HoverTool(tooltips=[
        ("Dataset", "@dataset"),
        ("Theta", "$x{0.000}"),
        ("Set Size", "$y{0}")
    ])
    p.add_tools(hover)
    return p


def setup_webdriver():
    """
    Set up and return a Firefox webdriver for Bokeh SVG export.
    """
    try:
        firefox_options = Options()
        firefox_options.add_argument("--headless")
        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)
        print("Successfully set up Firefox webdriver")
        return driver
    except Exception as e:
        print(f"Error setting up webdriver: {e}")
        return None


def save_plot_as_svg(plot, directory, filename, webdriver=None):
    """
    Save a Bokeh plot as an SVG file using the provided webdriver.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    export_svg(plot, filename=filepath, webdriver=webdriver)
    print(f"Saved plot as SVG: {filepath}")


def main():
    """
    Main function to run the dataset comparison and visualization.
    """
    svg_output_dir = "plots_dataset_comparison_rashomon"
    os.makedirs(svg_output_dir, exist_ok=True)
    print(f"Created directory for SVG exports: {svg_output_dir}")

    driver = setup_webdriver()

    experiment_dirs = find_experiment_dirs(args.directory, args.constant == 1)

    if not experiment_dirs:
        suffix_type = "with" if args.constant == 1 else "without"
        print(
            f"No experiment directories found {suffix_type} 'Constant' suffix in {args.directory}")
        return

    if args.verbose > 0:
        suffix_type = "with" if args.constant == 1 else "without"
        print(
            f"Found {len(experiment_dirs)} experiment directories {suffix_type} 'Constant' suffix:")
        for exp_dir in experiment_dirs:
            print(f"  - {exp_dir}")

    measures = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    titles = [m.replace('_', ' ').title() for m in measures]

    datasets_data = {}
    for exp_dir in experiment_dirs:
        dataset_name = os.path.basename(exp_dir)
        if args.verbose > 0:
            print(f"Loading {dataset_name}...")

        if args.fold == -1:

            df = load_all_folds_data(exp_dir)
        else:

            df = load_dataset_fold_data(exp_dir, args.fold)

        if df is not None:

            df = add_derived_metrics(df)

            df = df[df['recall'] > 0.0]
            if len(df) > 0:
                datasets_data[dataset_name] = df
            else:
                if args.verbose > 0:
                    print(
                        f"  No valid rule lists in {dataset_name} after filtering")
        else:
            if args.verbose > 0:
                print(f"  Could not load data for {dataset_name}")

    if not datasets_data:
        print("No valid datasets were loaded. Exiting.")
        return

    header = Div(text=f"""
    <div style="background-color: #2c3e50; color: white; padding: 15px; margin-bottom: 30px; border-radius: 5px; text-align: center;">
        <h1>Rashomon Sets: Cross-Dataset Comparison</h1>
        <p>Comparing Rashomon set sizes across different datasets</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, width_policy="max")

    measure_plots = []

    for measure, title in zip(measures, titles):
        if any(measure in df.columns for df in datasets_data.values()):

            show_all_folds = args.fold == -1

            raw_plot = create_dataset_comparison_plot(
                measure, title, datasets_data, show_all_folds=show_all_folds)
            norm_plot = create_normalized_dataset_comparison_plot(
                measure, title, datasets_data, show_all_folds=show_all_folds)

            safe_title = title.replace(" ", "_")
            norm_filename = f"{safe_title}_normalized.svg"
            save_plot_as_svg(norm_plot, svg_output_dir,
                             norm_filename, webdriver=driver)

            plot_header = Div(
                text=f"""<h2>{title} Rashomon Sets</h2>""", width_policy="max")

            combined_plots = row(raw_plot, norm_plot,
                                 sizing_mode="stretch_width")

            measure_section = column(
                plot_header,
                combined_plots,
                sizing_mode='stretch_width'
            )

            measure_plots.append(measure_section)
        else:
            if args.verbose > 0:
                print(f"Skipping {title} - no data found across datasets")

    layout = column(
        header,
        *measure_plots,
        sizing_mode='stretch_width'
    )

    suffix = "constant" if args.constant == 1 else "standard"
    output_filename = f"dataset_comparison_{suffix}_{args.output}"
    output_file(output_filename,
                title="Dataset Rashomon Comparison")
    save(layout)

    print(
        f"\nDataset comparison visualization created: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    main()
