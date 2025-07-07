import os
import argparse
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict


from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper, NumeralTickFormatter
from bokeh.palettes import Category10, Category20, viridis
from bokeh.transform import dodge
from bokeh.io import export_svg
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

parser = argparse.ArgumentParser(
    description="Analyze rule list features across datasets")
parser.add_argument("-d", "--directory", default="results_experiment2",
                    help="Directory containing experiment results (default: results_experiment2)")
parser.add_argument("-o", "--output", default="feature_analysis.html",
                    help="Output HTML file (default: feature_analysis.html)")
parser.add_argument("-d1", "--dataset1", default="Bank",
                    help="First dataset to analyze (default: Bank)")
parser.add_argument("-d2", "--dataset2", default="SUSY",
                    help="Second dataset to analyze (default: SUSY)")
parser.add_argument("-t", "--threshold", type=float, default=0.9,
                    help="Performance threshold for Rashomon set (default: 0.02)")
parser.add_argument("-v", "--verbose", type=int, default=1,
                    help="Verbose level (0-2) (default: 1)")
args = parser.parse_args()


output_dir = "results_feature_analysis"
os.makedirs(output_dir, exist_ok=True)


def find_dataset_dir(base_dir, dataset_name):
    """Find the directory for a specific dataset"""
    if args.verbose > 1:
        print(f"Looking for dataset directory: {dataset_name} in {base_dir}")
        print(f"Available directories: {os.listdir(base_dir)}")

    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path):

            if item.lower() == dataset_name.lower():

                fold_count = 0
                for i in range(5):
                    results_path = os.path.join(
                        full_path, f"fold_{i}", "results", "results_performance.csv")
                    if os.path.exists(results_path):
                        fold_count += 1

                if fold_count > 0:
                    if args.verbose > 0:
                        print(
                            f"Using dataset directory for {dataset_name}: {full_path} ({fold_count} folds)")
                    return full_path

    if args.verbose > 0:
        print(f"WARNING: Could not find directory for dataset {dataset_name}")
    return None


def load_dataset_rules(dataset_dir, measure='f1_score', threshold=0.02):
    """Load rule lists from all folds of a dataset"""
    if args.verbose > 1:
        print(f"Loading rules from {dataset_dir}")

    all_rules = []
    rashomon_rules = []
    best_performances = []

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

            best_performance = df[measure].max()
            best_performances.append(best_performance)

            threshold_value = best_performance - threshold

            df['in_rashomon'] = df[measure] >= threshold_value

            df['rule_length'] = df['rule'].apply(
                lambda x: len(x.split(', ')) - 1)

            all_rules.append(df)
            rashomon_rules.append(df[df['in_rashomon']])

        except Exception as e:
            if args.verbose > 0:
                print(f"Error loading {results_file}: {e}")

    if all_rules:
        all_df = pd.concat(all_rules, ignore_index=True)
        rashomon_df = pd.concat(rashomon_rules, ignore_index=True)
        if args.verbose > 0:
            print(
                f"Loaded {len(all_df)} total rules, {len(rashomon_df)} in Rashomon set")
        return all_df, rashomon_df, best_performances

    return None, None, []


def extract_features_from_rule(rule):
    """Extract individual features from a rule string"""

    features = re.findall(r'\{([^}]+)\}', rule)

    cleaned_features = []
    for feat in features:
        feat = feat.strip()
        if not feat.startswith('T='):
            cleaned_features.append(feat)

    return cleaned_features


def analyze_features(rules_df):
    """Analyze features in rule lists"""
    if args.verbose > 0:
        print(f"Analyzing features in {len(rules_df)} rule lists")

    feature_counts = Counter()

    first_feature_counts = Counter()

    feature_positions = defaultdict(list)

    feature_cooccurrence = defaultdict(Counter)

    for idx, row_data in rules_df.iterrows():
        rule_string = row_data['rule']

        rules = rule_string.split(', ')

        if idx < 2 and args.verbose > 1:
            print(f"\nSample rule list {idx}: {rule_string}")

        all_rule_features = []
        for i, rule in enumerate(rules):
            rule_features = extract_features_from_rule(rule)

            for feature in rule_features:

                feature_counts[feature] += 1

                feature_positions[feature].append(i + 1)

                if i == 0:
                    first_feature_counts[feature] += 1

                all_rule_features.append(feature)

        unique_features = set(all_rule_features)
        for feature1 in unique_features:
            for feature2 in unique_features:
                if feature1 != feature2:
                    feature_cooccurrence[feature1][feature2] += 1

    if args.verbose > 0:
        print(f"Found {len(feature_counts)} unique features")
        if feature_counts and args.verbose > 1:
            print(f"Top features: {feature_counts.most_common(5)}")

    return {
        'feature_counts': feature_counts,
        'first_feature_counts': first_feature_counts,
        'feature_positions': feature_positions,
        'feature_cooccurrence': feature_cooccurrence
    }


def create_feature_frequency_plot(dataset_name, analysis_results, measure, threshold):
    """Create feature frequency bar chart"""

    top_features = analysis_results['feature_counts'].most_common(15)

    if not top_features:
        if args.verbose > 0:
            print(f"No features found for {dataset_name}")
        return None

    features, counts = zip(*top_features)

    first_counts = [analysis_results['first_feature_counts'].get(
        f, 0) for f in features]

    p = figure(
        title=f"{dataset_name}: Most Common Features",
        height=400, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=features,
        y_axis_label="Frequency",
        toolbar_location="right"
    )

    source = ColumnDataSource(data=dict(
        features=features,
        counts=counts,
        first_counts=first_counts,

        first_percentage=[(fc/c)*100 if c > 0 else 0 for fc,
                          c in zip(first_counts, counts)]
    ))

    p.vbar(
        x='features', top='counts', source=source,
        width=0.7, fill_color="#1f77b4", line_color="white",
        fill_alpha=0.7, legend_label="Total Occurrences"
    )

    hover = HoverTool(tooltips=[
        ("Feature", "@features"),
        ("Total Occurrences", "@counts"),
        ("First Rule Occurrences", "@first_counts"),
        ("% as First Rule", "@first_percentage{0.0}%")
    ])
    p.add_tools(hover)

    p.xaxis.major_label_orientation = 3.14/4
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    return p


def create_feature_position_plot(dataset_name, analysis_results, measure, threshold):
    """Create feature position distribution plot"""

    top_features = analysis_results['feature_counts'].most_common(12)

    if not top_features:
        return None

    features = [f[0] for f in top_features]

    positions_data = {}

    for feature in features:
        if feature in analysis_results['feature_positions']:
            positions = analysis_results['feature_positions'][feature]

            position_counts = Counter(positions)
            positions_data[feature] = position_counts

    max_position = 0
    for feature, positions in positions_data.items():
        if positions:
            max_position = max(max_position, max(positions.keys()))

    p = figure(
        title=f"{dataset_name}: Feature Position Distribution",
        height=500, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="Rule Position",
        y_axis_label="Frequency",
        x_range=(0.5, max_position + 0.5),
        toolbar_location="right"
    )

    colors = Category10[10] if len(
        features) <= 10 else Category20[20][:len(features)]

    for i, feature in enumerate(features):
        if feature in positions_data and positions_data[feature]:

            positions = sorted(positions_data[feature].keys())
            counts = [positions_data[feature][pos] for pos in positions]

            source = ColumnDataSource(data=dict(
                positions=positions,
                counts=counts,
                feature=[feature] * len(positions)
            ))

            p.line(
                x='positions', y='counts', source=source,
                line_width=2, color=colors[i % len(colors)],
                legend_label=feature
            )

            p.scatter(
                x='positions', y='counts', source=source,
                size=8, color=colors[i % len(colors)]
            )

    hover = HoverTool(tooltips=[
        ("Feature", "@feature"),
        ("Position", "@positions"),
        ("Count", "@counts")
    ])
    p.add_tools(hover)

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    return p


def create_feature_cooccurrence_heatmap(dataset_name, analysis_results, measure, threshold):
    """Create feature co-occurrence heatmap"""

    full_features = [f[0]
                     for f in analysis_results['feature_counts'].most_common(15)]

    if not full_features:
        return None

    abbreviated = {}
    abbrev_counter = {}

    for feature in full_features:

        abbrev = feature[:3]

        if abbrev in abbreviated.values():

            if abbrev not in abbrev_counter:
                abbrev_counter[abbrev] = 1

            new_abbrev = f"{abbrev}{abbrev_counter[abbrev]}"
            abbrev_counter[abbrev] += 1
            abbreviated[feature] = new_abbrev
        else:
            abbreviated[feature] = abbrev

    top_features = [abbreviated[f] for f in full_features]

    cooccurrence_matrix = []

    for i, feature1 in enumerate(full_features):
        row_data = []
        for j, feature2 in enumerate(full_features):

            reversed_j = len(full_features) - 1 - j

            if feature1 == full_features[reversed_j]:

                row_data.append(float('nan'))
            else:

                row_data.append(
                    analysis_results['feature_cooccurrence'][feature1][feature2])
        cooccurrence_matrix.append(row_data)

    cooccurrence_array = np.array(cooccurrence_matrix)

    max_value = np.nanmax(cooccurrence_array)

    p = figure(
        title=f"{dataset_name}: Feature Co-occurrence Matrix",
        height=600, width=600,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=top_features,

        y_range=list(reversed(top_features)),
        toolbar_location="right"
    )

    mapper = LinearColorMapper(
        palette=viridis(256),
        low=0,
        high=max_value,
        nan_color="white"
    )

    heatmap_data = []
    for i, feature1 in enumerate(full_features):
        for j, feature2 in enumerate(full_features):
            reversed_j = len(full_features) - 1 - j

            actual_feature2 = full_features[reversed_j]

            if feature1 == actual_feature2:
                value = float('nan')
            else:
                value = analysis_results['feature_cooccurrence'][feature1][actual_feature2]

            is_diagonal = (feature1 == actual_feature2)

            heatmap_data.append({
                'feature1': abbreviated[feature1],
                'feature2': abbreviated[full_features[reversed_j]],
                'full_feature1': feature1,
                'full_feature2': full_features[reversed_j],
                'value': value,
                'is_diagonal': is_diagonal
            })

    source = ColumnDataSource(data=pd.DataFrame(heatmap_data))

    p.rect(
        x='feature1', y='feature2', width=1, height=1, source=source,
        fill_color={'field': 'value', 'transform': mapper},
        line_color="lightgray"
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        title="Co-occurrence Count",
        formatter=NumeralTickFormatter(format="0")
    )
    p.add_layout(color_bar, 'right')

    hover = HoverTool(tooltips=[
        ("Features", "@full_feature1 and @full_feature2"),
        ("Co-occurrences", "@value{0}"),
        ("Diagonal", "@is_diagonal")
    ])
    p.add_tools(hover)

    p.xaxis.major_label_orientation = 3.14/4

    return p


def create_rule_start_comparison(dataset1_name, dataset1_analysis, dataset2_name, dataset2_analysis, measure, threshold):
    """Create a comparison of starting features between datasets"""

    d1_starts = dataset1_analysis['first_feature_counts'].most_common(10)
    d2_starts = dataset2_analysis['first_feature_counts'].most_common(10)

    if not d1_starts or not d2_starts:
        return None

    all_start_features = list(
        set([f[0] for f in d1_starts] + [f[0] for f in d2_starts]))

    d1_counts = [dataset1_analysis['first_feature_counts'].get(
        f, 0) for f in all_start_features]
    d2_counts = [dataset2_analysis['first_feature_counts'].get(
        f, 0) for f in all_start_features]

    d1_total = sum(dataset1_analysis['first_feature_counts'].values())
    d2_total = sum(dataset2_analysis['first_feature_counts'].values())

    d1_percentages = [(count/d1_total)*100 if d1_total >
                      0 else 0 for count in d1_counts]
    d2_percentages = [(count/d2_total)*100 if d2_total >
                      0 else 0 for count in d2_counts]

    sorted_indices = np.argsort([max(d1_pct, d2_pct) for d1_pct, d2_pct in zip(
        d1_percentages, d2_percentages)])[::-1]

    sorted_features = [all_start_features[i] for i in sorted_indices]
    sorted_d1_percentages = [d1_percentages[i] for i in sorted_indices]
    sorted_d2_percentages = [d2_percentages[i] for i in sorted_indices]

    p = figure(
        title=f"Comparison of First Rule Features (Within {threshold} of best {measure}",
        height=500, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=sorted_features[:15],
        y_axis_label="Percentage of Rule Lists (%)",
        toolbar_location="right"
    )

    width = 0.35

    source = ColumnDataSource(data=dict(
        features=sorted_features[:15],
        d1_percentages=sorted_d1_percentages[:15],
        d2_percentages=sorted_d2_percentages[:15],
        d1_counts=d1_counts[:15],
        d2_counts=d2_counts[:15]
    ))

    p.vbar(
        x=dodge('features', -width/2, range=p.x_range), top='d1_percentages', source=source,
        width=width, fill_color="#1f77b4", line_color="white",
        fill_alpha=0.7, legend_label=dataset1_name
    )

    p.vbar(
        x=dodge('features', width/2, range=p.x_range), top='d2_percentages', source=source,
        width=width, fill_color="#ff7f0e", line_color="white",
        fill_alpha=0.7, legend_label=dataset2_name
    )

    hover = HoverTool(tooltips=[
        ("Feature", "@features"),
        (f"{dataset1_name} Percentage", "@d1_percentages{0.0}%"),
        (f"{dataset1_name} Count", "@d1_counts"),
        (f"{dataset2_name} Percentage", "@d2_percentages{0.0}%"),
        (f"{dataset2_name} Count", "@d2_counts")
    ])
    p.add_tools(hover)

    p.xaxis.major_label_orientation = 3.14/4
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    return p


def create_feature_performance_correlation(dataset_name, rules_df, analysis_results, measure, threshold):
    """Create plot showing correlation between feature presence and performance"""

    top_features = [f[0]
                    for f in analysis_results['feature_counts'].most_common(10)]

    if not top_features or len(rules_df) == 0:
        return None

    performance_data = []

    for feature in top_features:

        pattern = r'\{' + re.escape(feature) + r'\}'
        has_feature = rules_df['rule'].str.contains(pattern, regex=True)

        if has_feature.sum() > 0 and (~has_feature).sum() > 0:

            with_feature = rules_df[has_feature][measure]
            without_feature = rules_df[~has_feature][measure]

            performance_data.append({
                'feature': feature,
                'with_mean': with_feature.mean(),
                'without_mean': without_feature.mean(),
                'difference': with_feature.mean() - without_feature.mean(),
                'with_count': len(with_feature),
                'without_count': len(without_feature)
            })

    if not performance_data:
        return None

    performance_df = pd.DataFrame(performance_data).sort_values(
        'difference', ascending=False)

    p = figure(
        title=f"{dataset_name}: Feature Impact on {measure.capitalize()}",
        height=400, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=performance_df['feature'].tolist(),
        y_axis_label=f"Average {measure.capitalize()}",
        toolbar_location="right"
    )

    source = ColumnDataSource(performance_df)

    p.vbar(
        x=dodge('feature', -0.17, range=p.x_range), top='with_mean', source=source,
        width=0.3, fill_color="#1f77b4", line_color="white",
        fill_alpha=0.7, legend_label="With Feature"
    )

    p.vbar(
        x=dodge('feature', 0.17, range=p.x_range), top='without_mean', source=source,
        width=0.3, fill_color="#ff7f0e", line_color="white",
        fill_alpha=0.7, legend_label="Without Feature"
    )

    hover = HoverTool(tooltips=[
        ("Feature", "@feature"),
        ("With Feature", "@with_mean{0.000}"),
        ("Without Feature", "@without_mean{0.000}"),
        ("Difference", "@difference{+0.000}"),
        ("Rules With", "@with_count"),
        ("Rules Without", "@without_count")
    ])
    p.add_tools(hover)

    p.xaxis.major_label_orientation = 3.14/4
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.y_range.start = min(performance_df['with_mean'].min(
    ), performance_df['without_mean'].min()) * 0.95

    return p


def create_rule_length_comparison(dataset1_name, d1_rules_df, dataset2_name, d2_rules_df, measure, threshold):
    """Create comparison of rule lengths between datasets"""

    d1_lengths = d1_rules_df['rule_length'].value_counts().sort_index()
    d2_lengths = d2_rules_df['rule_length'].value_counts().sort_index()

    all_lengths = sorted(set(list(d1_lengths.index) + list(d2_lengths.index)))

    d1_counts = [d1_lengths.get(length, 0) for length in all_lengths]
    d2_counts = [d2_lengths.get(length, 0) for length in all_lengths]

    d1_total = sum(d1_counts)
    d2_total = sum(d2_counts)

    d1_percentages = [(count/d1_total)*100 if d1_total >
                      0 else 0 for count in d1_counts]
    d2_percentages = [(count/d2_total)*100 if d2_total >
                      0 else 0 for count in d2_counts]

    p = figure(
        title="Rule Length Distribution Comparison",
        height=400, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_axis_label="Rule Length (number of conditions)",
        y_axis_label="Percentage of Rule Lists (%)",
        toolbar_location="right"
    )

    source = ColumnDataSource(data=dict(
        lengths=all_lengths,
        d1_counts=d1_counts,
        d2_counts=d2_counts,
        d1_percentages=d1_percentages,
        d2_percentages=d2_percentages
    ))

    p.line(
        x='lengths', y='d1_percentages', source=source,
        line_width=3, color="#1f77b4", legend_label=dataset1_name
    )
    p.scatter(
        x='lengths', y='d1_percentages', source=source,
        size=8, color="#1f77b4"
    )

    p.line(
        x='lengths', y='d2_percentages', source=source,
        line_width=3, color="#ff7f0e", legend_label=dataset2_name
    )
    p.scatter(
        x='lengths', y='d2_percentages', source=source,
        size=8, color="#ff7f0e"
    )

    hover = HoverTool(tooltips=[
        ("Rule Length", "@lengths"),
        (f"{dataset1_name} Percentage", "@d1_percentages{0.0}%"),
        (f"{dataset1_name} Count", "@d1_counts"),
        (f"{dataset2_name} Percentage", "@d2_percentages{0.0}%"),
        (f"{dataset2_name} Count", "@d2_counts")
    ])
    p.add_tools(hover)

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    return p


def create_threshold_comparison_plot(dataset_name, all_rules_df, rashomon_rules_df, measure, threshold):
    """
    Create a plot comparing feature frequencies between Rashomon set and all rules
    """

    all_features_analysis = analyze_features(all_rules_df)
    rashomon_features_analysis = analyze_features(rashomon_rules_df)

    combined_counts = Counter()
    for feature, count in all_features_analysis['feature_counts'].items():
        combined_counts[feature] += count
    for feature, count in rashomon_features_analysis['feature_counts'].items():
        combined_counts[feature] += count

    top_features = [f[0] for f in combined_counts.most_common(15)]

    if not top_features:
        if args.verbose > 0:
            print(
                f"No features found for threshold comparison in {dataset_name}")
        return None

    all_counts = [all_features_analysis['feature_counts'].get(
        f, 0) for f in top_features]
    rashomon_counts = [rashomon_features_analysis['feature_counts'].get(
        f, 0) for f in top_features]

    all_total = sum(all_features_analysis['feature_counts'].values()) or 1
    rashomon_total = sum(
        rashomon_features_analysis['feature_counts'].values()) or 1

    all_percentages = [(count/all_total)*100 for count in all_counts]
    rashomon_percentages = [(count/rashomon_total) *
                            100 for count in rashomon_counts]

    diff_percentages = [r - a for r,
                        a in zip(rashomon_percentages, all_percentages)]

    sorted_indices = np.argsort([abs(diff) for diff in diff_percentages])[::-1]

    sorted_features = [top_features[i] for i in sorted_indices]
    sorted_all_percentages = [all_percentages[i] for i in sorted_indices]
    sorted_rashomon_percentages = [
        rashomon_percentages[i] for i in sorted_indices]
    sorted_diff_percentages = [diff_percentages[i] for i in sorted_indices]

    bar_colors = ["#1f77b4" if x >=
                  0 else "#d62728" for x in sorted_diff_percentages[:15]]

    p = figure(
        title=f"{dataset_name}: Feature Frequency Comparison",
        height=500, width=800,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=sorted_features[:15],
        y_axis_label="Percentage Difference (%)",
        toolbar_location="right"
    )

    source = ColumnDataSource(data=dict(
        features=sorted_features[:15],
        all_percentages=sorted_all_percentages[:15],
        rashomon_percentages=sorted_rashomon_percentages[:15],
        diff_percentages=sorted_diff_percentages[:15],
        bar_colors=bar_colors,
        all_counts=[all_features_analysis['feature_counts'].get(
            f, 0) for f in sorted_features[:15]],
        rashomon_counts=[rashomon_features_analysis['feature_counts'].get(
            f, 0) for f in sorted_features[:15]]
    ))

    p.vbar(
        x='features', top='diff_percentages', source=source,
        width=0.7,
        fill_color='bar_colors',
        line_color="white",
        fill_alpha=0.7
    )

    p.line(

        x=list(range(len(sorted_features[:15]))), y=[0]*len(sorted_features[:15]),
        line_width=1, color="black", line_dash="dashed"
    )

    hover = HoverTool(tooltips=[
        ("Feature", "@features"),
        ("Rashomon Set %", "@rashomon_percentages{0.0}%"),
        ("All Rules %", "@all_percentages{0.0}%"),
        ("Difference", "@diff_percentages{+0.0}%"),
        ("Rashomon Count", "@rashomon_counts"),
        ("All Rules Count", "@all_counts")
    ])
    p.add_tools(hover)

    p.xaxis.major_label_orientation = 3.14/4

    return p


def save_plot_svg(plot, filename, webdriver=None):
    """Save a Bokeh plot as an SVG file using the provided webdriver"""
    plot.output_backend = "svg"
    svg_path = os.path.join(output_dir, filename)
    try:
        export_svg(plot, filename=svg_path, webdriver=webdriver)
        print(f"Saved SVG: {svg_path}")
    except Exception as e:
        print(f"Failed to save SVG for {filename}: {e}")
    return svg_path


def setup_webdriver():
    """Set up and return a Firefox webdriver for Bokeh SVG export"""
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


def analyze_and_save_for_dataset(dataset_name, dataset_dir, webdriver=None):

    dataset_name = dataset_name.replace("_", "").replace(
        "Constant", "").replace("reduced", "")
    all_rules, rashomon_rules, best_performances = load_dataset_rules(
        dataset_dir, 'f1_score', args.threshold)
    if rashomon_rules is None:
        print(f"No Rashomon rules for {dataset_name}")
        return
    analysis = analyze_features(rashomon_rules)

    plots = []

    freq_plot = create_feature_frequency_plot(
        dataset_name, analysis, 'f1_score', args.threshold)
    co_plot = create_feature_cooccurrence_heatmap(
        dataset_name, analysis, 'f1_score', args.threshold)
    pos_plot = create_feature_position_plot(
        dataset_name, analysis, 'f1_score', args.threshold)
    perf_plot = create_feature_performance_correlation(
        dataset_name, rashomon_rules, analysis, 'f1_score', args.threshold)
    length_plot = create_rule_length_comparison(
        dataset_name, rashomon_rules, dataset_name, rashomon_rules, 'f1_score', args.threshold)
    thresh_plot = create_threshold_comparison_plot(
        dataset_name, all_rules, rashomon_rules, 'f1_score', args.threshold)
    if freq_plot:
        output_file(os.path.join(
            output_dir, f"{dataset_name}_feature_frequency.html"))
        save(freq_plot)
        save_plot_svg(
            freq_plot, f"{dataset_name}_feature_frequency.svg", webdriver)
        plots.append(freq_plot)

    if pos_plot:
        output_file(os.path.join(
            output_dir, f"{dataset_name}_feature_position.html"))
        save(pos_plot)
        save_plot_svg(
            pos_plot, f"{dataset_name}_feature_position.svg", webdriver)
        plots.append(pos_plot)
    if co_plot:
        output_file(os.path.join(
            output_dir, f"{dataset_name}_cooccurrence_heatmap.html"))
        save(co_plot)
        save_plot_svg(
            co_plot, f"{dataset_name}_cooccurrence_heatmap.svg", webdriver)
        plots.append(co_plot)
    if perf_plot:
        output_file(os.path.join(
            output_dir, f"{dataset_name}_performance_correlation.html"))
        save(perf_plot)
        save_plot_svg(
            perf_plot, f"{dataset_name}_performance_correlation.svg", webdriver)
        plots.append(perf_plot)
    if length_plot:
        output_file(os.path.join(
            output_dir, f"{dataset_name}_rule_length_comparison.html"))
        save(length_plot)
        save_plot_svg(
            length_plot, f"{dataset_name}_rule_length_comparison.svg", webdriver)
        plots.append(length_plot)
    if thresh_plot:
        output_file(os.path.join(
            output_dir, f"{dataset_name}_threshold_comparison.html"))
        save(thresh_plot)
        save_plot_svg(
            thresh_plot, f"{dataset_name}_threshold_comparison.svg", webdriver)
        plots.append(thresh_plot)
    print(f"Finished plots for {dataset_name}")


def main():
    webdriver = setup_webdriver()

    base_dir = args.directory
    for dataset_folder in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_folder)
        if os.path.isdir(dataset_path):
            analyze_and_save_for_dataset(
                dataset_folder, dataset_path, webdriver)
    print("All dataset plots generated and saved as HTML and SVG.")


if __name__ == "__main__":
    main()
