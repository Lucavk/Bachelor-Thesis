import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


from bokeh.plotting import figure, output_file, save
from bokeh.models import (ColumnDataSource, HoverTool, Tabs, TabPanel,
                          Legend, LegendItem, Div)
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.io import export_svg

from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

parser = argparse.ArgumentParser(
    description="Visualize Rashomon sets across multiple folds")
parser.add_argument("-d", "--directory", default="results_experiment2",
                    help="Directory containing experiment results (default: results_experiments)")
parser.add_argument("-o", "--output", default="rashomon_visualization.html",
                    help="Output HTML file (default: rashomon_visualization.html)")
parser.add_argument("-constant", type=int, default=0,
                    help="Process folders ending with 'Constant' (1) or not (0) (default: 0)")
parser.add_argument("-v", "--verbose", type=int, default=1,
                    help="Verbose level (0-2) (default: 1)")
args = parser.parse_args()


def find_experiment_dirs(base_dir="results_experiment2", constant_suffix=False):
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

                has_folds_with_results = True
                for i in range(5):
                    fold_path = os.path.join(full_path, f"fold_{i}")
                    results_file = os.path.join(
                        fold_path, "results", "results_performance.csv")
                    if not (os.path.isdir(fold_path) and os.path.exists(results_file)):
                        has_folds_with_results = False
                        break

                if has_folds_with_results:
                    experiment_dirs.append(full_path)

    return experiment_dirs


def load_fold_data(experiment_dir, fold_num):
    """
    Load data from a specific fold in an experiment.
    """
    results_file = os.path.join(
        experiment_dir, f"fold_{fold_num}", "results", "results_performance.csv")

    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file, sep=';')

            df['fold'] = fold_num

            if args.verbose > 1:
                print(f"  Loaded {len(df)} rules from fold_{fold_num}")

            return df
        except Exception as e:
            if args.verbose > 0:
                print(f"  Error loading {results_file}: {e}")
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


def create_rashomon_size_plot(dataset_name, measure, title, fold_data_list):
    """
    Create Rashomon set size plot for a specific dataset and performance measure.
    """
    p = figure(
        title=f"{dataset_name}: Rashomon Set Size for {title}",
        height=400, width=700,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="θ (threshold)",
        y_axis_label="Rashomon Set Size |R(θ)|"
    )

    colors = Category10[10][:5]

    legend_items = []

    all_thetas = np.linspace(0, 0.2, 100)
    sum_sizes = np.zeros(len(all_thetas))
    fold_count = 0

    for i, fold_df in enumerate(fold_data_list):
        if fold_df is None or fold_df.empty:
            continue

        fold_count += 1
        total_rules = len(fold_df)

        best_performance = fold_df[measure].max()

        theta_values = np.linspace(0, 0.2, 100)

        rashomon_sizes = []
        threshold_values = []

        for j, theta in enumerate(theta_values):

            threshold = best_performance - theta
            threshold_values.append(threshold)

            rashomon_size = len(fold_df[fold_df[measure] >= threshold])
            rashomon_sizes.append(rashomon_size)

            sum_sizes[j] += rashomon_size

        source = ColumnDataSource(data=dict(
            theta=theta_values,
            size=rashomon_sizes,
            threshold=threshold_values,
            fold=[f"fold_{i}"] * len(theta_values),
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
            label=f"Fold {i} ({total_rules} rules)", renderers=[line]))

    hover = HoverTool(tooltips=[
        ("Fold", "@fold"),
        ("Theta", "@theta{0.000}"),
        ("Set Size", "@size{0}"),
        ("Total Rules", "@total_rules"),
        ("Performance Threshold", "@threshold{0.000}"),
        ("Best Performance", "@best_performance{0.000}")
    ], mode="vline")

    p.add_tools(hover)

    legend = Legend(items=legend_items, location="center")
    p.add_layout(legend, 'right')

    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"

    return p


def create_normalized_rashomon_plot(dataset_name, measure, title, fold_data_list):
    """
    Create normalized Rashomon set size plot (as percentage of total rule lists).
    """
    dataset_name = dataset_name.replace("_", "").replace(
        "Constant", "").replace("reduced", "")
    p = figure(
        title=f"{dataset_name}: Normalized Rashomon Set for {title}",
        height=400, width=700,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="threshold θ",
        y_axis_label="size of Rashomon set (percentage of all rules)",
        y_range=(0, 100)
    )

    colors = Category10[10][:5]
    legend_items = []

    all_thetas = np.linspace(0, 0.2, 100)
    sum_percentages = np.zeros(len(all_thetas))
    fold_count = 0

    for i, fold_df in enumerate(fold_data_list):
        if fold_df is None or fold_df.empty:
            continue

        fold_count += 1
        best_performance = fold_df[measure].max()
        total_rules = len(fold_df)

        theta_values = np.linspace(0, 0.2, 100)

        norm_sizes = []
        threshold_values = []
        abs_sizes = []

        for j, theta in enumerate(theta_values):
            threshold = best_performance - theta
            threshold_values.append(threshold)
            rashomon_size = len(fold_df[fold_df[measure] >= threshold])
            abs_sizes.append(rashomon_size)

            norm_size = (rashomon_size / total_rules *
                         100) if total_rules > 0 else 0
            norm_sizes.append(norm_size)

            sum_percentages[j] += norm_size

        source = ColumnDataSource(data=dict(
            theta=theta_values,
            size=norm_sizes,
            abs_size=abs_sizes,
            threshold=threshold_values,
            fold=[f"fold_{i + 1}"] * len(theta_values),
            total_rules=[total_rules] * len(theta_values),
            best_performance=[best_performance] * len(theta_values)
        ))

        line = p.line('theta', 'size', source=source,
                      line_width=2.5, line_color=colors[i], alpha=0.8)

        p.scatter('theta', 'size', source=source,
                  size=4, fill_color="white", line_color=colors[i], alpha=0.6)

        legend_items.append(LegendItem(
            label=f"Fold {i + 1} ({total_rules} rule lists)", renderers=[line]))

    hover = HoverTool(tooltips=[
        ("Fold", "@fold"),
        ("Theta", "@theta{0.000}"),
        ("% of Rules", "@size{0.0}%"),
        ("Absolute Size", "@abs_size{0}"),
        ("Total Rules", "@total_rules"),
        ("Performance Threshold", "@threshold{0.000}"),
        ("Best Performance", "@best_performance{0.000}")
    ], mode="vline")

    p.add_tools(hover)

    legend = Legend(items=legend_items, location="center")
    p.add_layout(legend, 'right')

    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"

    return p


def save_plot_as_pdf(plot, directory, filename, driver=None):
    """
    Save a Bokeh plot as a PDF file with high resolution.
    """

    os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, filename)

    svg_filename = filepath.replace('.pdf', '.svg')
    export_svg(plot, filename=svg_filename,
               webdriver=driver)

    print(f"Saved plot as SVG: {filepath}")


def setup_webdriver():
    """
    Set up and return a webdriver for Bokeh exports.
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


def main():
    """
    Main function to run the folds comparison and visualization.
    """
    pdf_output_dir = "plots_folds_comparison_rashomon"
    os.makedirs(pdf_output_dir, exist_ok=True)
    print(f"Created directory for PDF exports: {pdf_output_dir}")

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

    all_plots = []

    for exp_dir in experiment_dirs:
        dataset_name = os.path.basename(exp_dir)
        if args.verbose > 0:
            print(f"Processing {dataset_name}...")

        fold_data = []
        for fold_num in range(5):
            fold_df = load_fold_data(exp_dir, fold_num)
            if fold_df is not None:

                fold_df = add_derived_metrics(fold_df)

                fold_df = fold_df[fold_df['recall'] > 0.0]
                fold_data.append(fold_df)
            else:
                fold_data.append(None)

        dataset_header = Div(text=f"""
        <div style="background-color: #2c3e50; color: white; padding: 10px; margin-bottom: 20px; border-radius: 5px;">
            <h2>Dataset: {dataset_name}</h2>
            <p>Visualization of Rashomon sets across 5 folds</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, width_policy="max")

        dataset_plots = []
        for measure, title in zip(measures, titles):
            if any(fold_df is not None and measure in fold_df.columns for fold_df in fold_data):

                raw_plot = create_rashomon_size_plot(
                    dataset_name, measure, title, fold_data)
                norm_plot = create_normalized_rashomon_plot(
                    dataset_name, measure, title, fold_data)

                safe_dataset_name = dataset_name.replace(
                    " ", "_").replace("/", "_")

                safe_title = title.replace(" ", "_")

                norm_filename = f"{safe_dataset_name}_{safe_title}_normalized.pdf"
                save_plot_as_pdf(norm_plot, pdf_output_dir,
                                 norm_filename, driver=driver)

                plot_header = Div(
                    text=f"""<h3>{title} Rashomon Sets</h3>""", width_policy="max")

                combined_plots = column(
                    plot_header,
                    row(raw_plot, norm_plot, sizing_mode='stretch_width'),
                    sizing_mode='stretch_width'
                )

                dataset_plots.append(combined_plots)
            else:
                if args.verbose > 0:
                    print(
                        f"  Skipping {title} plot - measure not found in data")

        rashomon_explanation = Div(text="""
        <div style="background-color: #eaf2f8; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; border-radius: 0 5px 5px 0;">
            <h3>About Rashomon Sets</h3>
            <p>A Rashomon set R(θ) consists of all models that perform within θ of the best-performing model.
            Formally, for a performance measure P the Rashomon set is defined as:
            R(θ) = {model m : Performance(m) ≥ Performance(best) - θ}</p>
            <p>Larger Rashomon sets indicate more diverse models with similar performance.</p>
            <p><strong>Left plots</strong> show the absolute size of Rashomon sets (number of rules).</p>
            <p><strong>Right plots</strong> show the normalized size (percentage of total rules in each fold).</p>
        </div>
        """, width_policy="max")

        fold_explanation = Div(text="""
        <div style="background-color: #fef9e7; border-left: 4px solid #f1c40f; padding: 15px; margin: 20px 0; border-radius: 0 5px 5px 0;">
            <h3>Understanding Fold Differences</h3>
            <p>Differences in Rashomon set sizes between folds can be attributed to:</p>
            <ul>
                <li>Different data distributions across folds</li>
                <li>Varying number of rules discovered per fold</li>
                <li>Differences in the best performance achieved in each fold</li>
            </ul>
            <p>The normalized view (right plots) helps to compare the relative sizes despite these differences.</p>
        </div>
        """, width_policy="max")

        grid = column(dataset_plots, sizing_mode='stretch_width')

        dataset_content = column(
            dataset_header,


            grid)
        all_plots.append((dataset_name, dataset_content))

    tabs = []
    for dataset_name, dataset_content in all_plots:
        tab = TabPanel(child=dataset_content, title=dataset_name)
        tabs.append(tab)

    tabbed_layout = Tabs(tabs=tabs)

    suffix = "constant" if args.constant == 1 else "standard"
    output_filename = f"rashomon_{suffix}_{args.output}"
    output_file(output_filename,
                title=f"Rashomon Set Visualization - {suffix.capitalize()} Experiments")
    save(tabbed_layout)

    print(
        f"\nRashomon set visualization created: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    main()
