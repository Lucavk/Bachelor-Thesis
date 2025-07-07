import os
import argparse
import pandas as pd


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from bokeh.io import export_svg
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options


parser = argparse.ArgumentParser(
    description="Analyze rule list complexity across datasets")
parser.add_argument("-d", "--directory", default="results_experiment3",
                    help="Directory containing experiment results (default: results_experiment3)")
parser.add_argument("-v", "--verbose", type=int, default=1,
                    help="Verbose level (0-2) (default: 1)")
parser.add_argument("-m", "--measure", default="f1_score",
                    help="Performance measure for comparison plot (default: f1_score)")
args = parser.parse_args()


output_dir = "results_complexity_analysis"
os.makedirs(output_dir, exist_ok=True)
if args.verbose > 0:
    print(f"Created output directory: {output_dir}")


def setup_webdriver():
    """Set up and return a Firefox webdriver for Bokeh SVG export"""
    try:
        firefox_options = Options()
        firefox_options.add_argument("--headless")
        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)
        if args.verbose > 0:
            print("Successfully set up Firefox webdriver")
        return driver
    except Exception as e:
        if args.verbose > 0:
            print(f"Error setting up webdriver: {e}")
        return None


def save_plot_svg(plot, filename, webdriver=None):
    """Save a Bokeh plot as an SVG file using the provided webdriver"""
    plot.output_backend = "svg"
    svg_path = os.path.join(output_dir, filename)
    try:
        export_svg(plot, filename=svg_path, webdriver=webdriver)
        if args.verbose > 0:
            print(f"Saved SVG: {svg_path}")
    except Exception as e:
        if args.verbose > 0:
            print(f"Failed to save SVG for {filename}: {e}")
    return svg_path


def load_dataset_rules(dataset_dir):
    """Load rule lists from all folds of a dataset"""
    if args.verbose > 1:
        print(f"Loading rules from {dataset_dir}")

    all_rules = []

    for fold_num in range(5):
        results_file = os.path.join(
            dataset_dir, f"fold_{fold_num}", "results", "results_performance.csv")

        if not os.path.exists(results_file):
            if args.verbose > 0:
                print(f"Results file not found: {results_file}")
            continue

        try:
            df = pd.read_csv(results_file, sep=';')

            if args.verbose > 1:
                print(f"Loaded fold {fold_num} with {len(df)} rules")
                if not df.empty:
                    print(f"Sample rule: {df['rule'].iloc[0]}")

            df['fold'] = fold_num

            df['rule_complexity'] = df['rule'].apply(
                lambda x: len(x.split(', ')) - 1)

            all_rules.append(df)

        except Exception as e:
            if args.verbose > 0:
                print(f"Error loading {results_file}: {e}")

    if all_rules:
        all_df = pd.concat(all_rules, ignore_index=True)
        if args.verbose > 0:
            print(f"Loaded {len(all_df)} total rules")
        return all_df

    return None


def create_complexity_performance_plot(dataset_name, df, measure):
    """Create a scatter plot of rule complexity vs. performance for a specific measure"""
    if measure not in df.columns:
        if args.verbose > 0:
            print(
                f"Performance measure {measure} not found in dataset {dataset_name}")
        return None

    display_name = dataset_name.replace("_", " ").replace(
        "Constant", "").replace("reduced", "")

    p = figure(
        title=f"{display_name}: Rule Complexity vs. {measure.replace('_', ' ').title()}",
        height=500, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="Rule Complexity (number of rules)",
        y_axis_label=f"{measure.replace('_', ' ').title()}",
        toolbar_location="right"
    )

    complexity_stats = df.groupby('rule_complexity')[measure].agg(
        ['mean', 'std', 'count', 'min', 'max']).reset_index()

    complexity_stats['lower_bound'] = complexity_stats['mean'] - \
        complexity_stats['std']
    complexity_stats['upper_bound'] = complexity_stats['mean'] + \
        complexity_stats['std']

    avg_source = ColumnDataSource(data=dict(
        complexity=complexity_stats['rule_complexity'],
        performance=complexity_stats['mean'],
        std=complexity_stats['std'],
        count=complexity_stats['count'],
        min_perf=complexity_stats['min'],
        max_perf=complexity_stats['max'],
        lower_bound=complexity_stats['lower_bound'],
        upper_bound=complexity_stats['upper_bound']
    ))

    point_source = ColumnDataSource(data=dict(
        complexity=df['rule_complexity'],
        performance=df[measure],
        fold=df['fold'].astype(str),
        rule=[r[:50] + "..." if len(r) > 50 else r for r in df['rule']]
    ))

    scatter = p.scatter(
        x='complexity', y='performance', source=point_source,
        size=6, color="#1f77b4", alpha=0.2,
        legend_label="Individual Rule Lists"
    )

    line = p.line(
        x='complexity', y='performance', source=avg_source,
        line_width=3, color="#ff7f0e",
        legend_label="Average Performance"
    )

    p.segment(
        x0='complexity', y0='lower_bound', x1='complexity', y1='upper_bound',
        source=avg_source, line_width=2, line_color="#ff7f0e", line_alpha=0.5
    )

    point_hover = HoverTool(tooltips=[
        ("Complexity", "@complexity"),
        (f"{measure.title()}", "@performance{0.000}"),
        ("Fold", "@fold"),
        ("Rule", "@rule")
    ], renderers=[scatter])

    avg_hover = HoverTool(tooltips=[
        ("Complexity", "@complexity"),
        (f"Avg {measure.title()}", "@performance{0.000}"),
        ("Std Dev", "@std{0.000}"),
        ("Count", "@count"),
        ("Min", "@min_perf{0.000}"),
        ("Max", "@max_perf{0.000}")
    ], renderers=[line])

    p.add_tools(point_hover, avg_hover)

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"

    return p


def create_datasets_best_comparison_plot(datasets_data, measure):
    """Create a comparison plot of best complexity vs. performance across datasets"""

    p = figure(
        title=f"Best Rule Complexity vs. {measure.replace('_', ' ').title()} Across Datasets",
        height=600, width=1000,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="Rule Complexity (number of rules)",
        y_axis_label=f"{measure.replace('_', ' ').title()}",
        toolbar_location="right"
    )

    colors = Category10[10]

    for i, (dataset_name, df) in enumerate(datasets_data.items()):
        if measure not in df.columns:
            continue

        display_name = dataset_name.replace("_", " ").replace(
            "Constant", "").replace("reduced", "")

        complexity_stats = df.groupby('rule_complexity')[
            measure].max().reset_index()

        source = ColumnDataSource(data=dict(
            complexity=complexity_stats['rule_complexity'],
            performance=complexity_stats[measure],
            dataset=[display_name] * len(complexity_stats)
        ))

        line = p.line(
            x='complexity', y='performance', source=source,
            line_width=3, color=colors[i % len(colors)],
            legend_label=display_name
        )

        p.scatter(
            x='complexity', y='performance', source=source,
            size=8, color=colors[i % len(colors)]
        )

    hover = HoverTool(tooltips=[
        ("Dataset", "@dataset"),
        ("Complexity", "@complexity"),
        (f"Best {measure.title()}", "@performance{0.000}")
    ])
    p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def create_datasets_comparison_plot(datasets_data, measure):
    """Create a comparison plot of complexity vs. performance across datasets"""

    p = figure(
        title=f"Comparison of Rule Complexity vs. {measure.replace('_', ' ').title()} Across Datasets",
        height=600, width=1000,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="Rule Complexity (number of rules)",
        y_axis_label=f"{measure.replace('_', ' ').title()}",
        toolbar_location="right"
    )

    colors = Category10[10]

    for i, (dataset_name, df) in enumerate(datasets_data.items()):
        if measure not in df.columns:
            continue

        display_name = dataset_name.replace("_", " ").replace(
            "Constant", "").replace("reduced", "")

        complexity_stats = df.groupby('rule_complexity')[
            measure].mean().reset_index()

        source = ColumnDataSource(data=dict(
            complexity=complexity_stats['rule_complexity'],
            performance=complexity_stats[measure],
            dataset=[display_name] * len(complexity_stats)
        ))

        line = p.line(
            x='complexity', y='performance', source=source,
            line_width=3, color=colors[i % len(colors)],
            legend_label=display_name
        )

        p.scatter(
            x='complexity', y='performance', source=source,
            size=8, color=colors[i % len(colors)]
        )

    hover = HoverTool(tooltips=[
        ("Dataset", "@dataset"),
        ("Complexity", "@complexity"),
        (f"{measure.title()}", "@performance{0.000}")
    ])
    p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def main():
    webdriver = setup_webdriver()

    measures = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']

    datasets_data = {}

    base_dir = args.directory
    if args.verbose > 0:
        print(f"Processing datasets from {base_dir}")

    for dataset_folder in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        if args.verbose > 0:
            print(f"Processing dataset: {dataset_folder}")

        df = load_dataset_rules(dataset_path)
        if df is None:
            continue

        datasets_data[dataset_folder] = df

        for measure in measures:
            if measure in df.columns:
                plot = create_complexity_performance_plot(
                    dataset_folder, df, measure)
                if plot:

                    safe_name = dataset_folder.replace(
                        " ", "_").replace("/", "_")
                    filename = f"{safe_name}_{measure}_complexity.svg"
                    save_plot_svg(plot, filename, webdriver)

    for measure in measures:
        if args.verbose > 0:
            print(
                f"Creating average cross-dataset comparison plot for {measure} (Experiment 3)...")
        comp_plot = create_datasets_comparison_plot(
            datasets_data, measure)
        if comp_plot:
            filename = f"datasets_comparison_avg_{measure}_complexity.svg"
            save_plot_svg(comp_plot, filename, webdriver)

        if args.verbose > 0:
            print(
                f"Creating best cross-dataset comparison plot for {measure} (Experiment 3)...")
        best_comp_plot = create_datasets_best_comparison_plot(
            datasets_data, measure)
        if best_comp_plot:
            filename = f"datasets_comparison_best_{measure}_complexity.svg"
            save_plot_svg(best_comp_plot, filename, webdriver)

    experiment2_dir = "results_experiment2"
    if os.path.exists(experiment2_dir) and os.path.isdir(experiment2_dir):
        if args.verbose > 0:
            print(
                f"\nProcessing datasets from {experiment2_dir} for f1_score comparison")

        experiment2_datasets_data = {}

        for dataset_folder in os.listdir(experiment2_dir):
            dataset_path = os.path.join(experiment2_dir, dataset_folder)
            if not os.path.isdir(dataset_path):
                continue

            if args.verbose > 0:
                print(f"Processing dataset: {dataset_folder}")

            df = load_dataset_rules(dataset_path)
            if df is None or 'f1_score' not in df.columns:
                continue

            experiment2_datasets_data[dataset_folder] = df

        if experiment2_datasets_data:
            if args.verbose > 0:
                print(
                    "Creating average cross-dataset comparison plot for f1_score (Experiment 2)...")
            comp_plot = create_datasets_comparison_plot(
                experiment2_datasets_data, 'f1_score')
            if comp_plot:
                filename = "experiment2_datasets_comparison_avg_f1_score_complexity.svg"
                save_plot_svg(comp_plot, filename, webdriver)

            if args.verbose > 0:
                print(
                    "Creating best cross-dataset comparison plot for f1_score (Experiment 2)...")
            best_comp_plot = create_datasets_best_comparison_plot(
                experiment2_datasets_data, 'f1_score')
            if best_comp_plot:
                filename = "experiment2_datasets_comparison_best_f1_score_complexity.svg"
                save_plot_svg(best_comp_plot, filename, webdriver)

    if args.verbose > 0:
        print("Complexity analysis complete. SVG plots saved to", output_dir)


if __name__ == "__main__":
    main()
