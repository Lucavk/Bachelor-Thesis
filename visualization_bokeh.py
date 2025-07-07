import pandas as pd
import numpy as np
import argparse
import os
import random
import re
from datetime import datetime


from bokeh.plotting import figure, output_file, save
from bokeh.models import (ColumnDataSource, HoverTool, ColorBar, LinearColorMapper,
                          BasicTicker, Div, Range1d, CrosshairTool,)
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import viridis, plasma, Blues
from bokeh.transform import linear_cmap
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter


parser = argparse.ArgumentParser()
parser.add_argument("-fn", help="FileName (csv file), e.g. 'results_performance.csv'",
                    default="results_performance.csv")
parser.add_argument("-o", help="Output file path for dashboard HTML",
                    default="rule_dashboard_bokeh.html")
parser.add_argument("-v", help="Verbose level (0-2)", type=int, default=1)
parser.add_argument("-min_recall", help="Minimum recall to include (filter trivial rules)",
                    type=float, default=0.01)
args = parser.parse_args()

if args.v > 0:
    print(f"Loading data from {args.fn}")


os.makedirs(os.path.dirname(os.path.abspath(args.o)) or '.', exist_ok=True)


df_metrics = pd.read_csv(args.fn, sep=';')


df_metrics['rule_length'] = df_metrics['rule'].apply(
    lambda x: len(x.split(', ')) - 1)


if 'is_exact' in df_metrics.columns:
    df_metrics['is_exact'] = df_metrics['is_exact'].astype(bool)
else:
    df_metrics['is_exact'] = False


df_filtered = df_metrics[df_metrics['recall'] >= args.min_recall].copy()


df_filtered['false_positive_rate'] = df_filtered['fp'] / \
    (df_filtered['fp'] + df_filtered['tn'])
df_filtered['true_positive_rate'] = df_filtered['recall']


df_filtered['solution_type'] = df_filtered['is_exact'].apply(
    lambda x: 'Exact' if x else 'Approximate')

if args.v > 0:
    print(f"Loaded {len(df_metrics)} unique rules")
    print(
        f"Filtered to {len(df_filtered)} rules with recall >= {args.min_recall}")
    if 'is_exact' in df_filtered.columns:
        exact_count = df_filtered['is_exact'].sum()
        print(f"Exact solutions: {exact_count}")
        print(f"Approximate solutions: {len(df_filtered) - exact_count}")


feature_counts = {}
for rule in df_filtered['rule']:
    features = re.findall(r'\{([^}]+)\}', rule)
    filtered_features = [f for f in features if f != "T=0" and f != "T=1"]

    for feature in filtered_features:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1

top_features = sorted(feature_counts.items(),
                      key=lambda x: x[1], reverse=True)[:10]


output_file(args.o, title="Rule Visualization Dashboard")


source = ColumnDataSource(df_filtered)


rule_tooltips = [
    ("Rule", "@rule"),
    ("Accuracy", "@accuracy{0.000}"),
    ("F1 Score", "@f1_score{0.000}"),
    ("Precision", "@precision{0.000}"),
    ("Recall", "@recall{0.000}"),
    ("Rule Length", "@rule_length"),
    ("Avg K", "@avg_k{0.00}"),
    ("Avg Z", "@avg_z{0.00}"),
    ("Solution Type", "@solution_type")
]


colors = {'Exact': '#FF4500', 'Approximate': '#3498DB'}


def create_rule_length_plot():

    mapper = LinearColorMapper(palette=viridis(256),
                               low=df_filtered['f1_score'].min(),
                               high=df_filtered['f1_score'].max())

    color_bar = ColorBar(color_mapper=mapper,
                         title="F1 Score",
                         location=(0, 0),
                         ticker=BasicTicker())

    p1 = figure(title="Rule Length vs Performance",
                height=400, width=650,
                tools="pan,box_zoom,wheel_zoom,reset,save",
                x_axis_label="Rule Length (number of conditions)",
                y_axis_label="Accuracy")

    exact_source = ColumnDataSource(df_filtered[df_filtered['is_exact']])
    approx_source = ColumnDataSource(df_filtered[~df_filtered['is_exact']])

    approx_scatter = p1.scatter('rule_length', 'accuracy', source=approx_source,
                                size=10, alpha=0.7,
                                line_color="black", line_width=0.5,
                                color=linear_cmap('f1_score', viridis(256),
                                                  df_filtered['f1_score'].min(
                                ),
                                    df_filtered['f1_score'].max()),
                                legend_label='Approximate')

    exact_scatter = p1.scatter('rule_length', 'accuracy', source=exact_source,
                               size=14, alpha=0.9,
                               marker='diamond',
                               line_color="black", line_width=1.0,
                               color='#FF4500',
                               legend_label='Exact')

    hover = HoverTool(tooltips=rule_tooltips)
    hover.attachment = "vertical"
    hover.point_policy = "follow_mouse"

    p1.add_tools(hover)
    p1.add_layout(color_bar, 'right')

    p1.legend.location = "top_left"
    p1.legend.click_policy = "hide"

    return p1


def create_precision_recall_plot():
    p2 = figure(title="Precision-Recall Trade-off",
                height=400, width=650,
                tools="pan,box_zoom,wheel_zoom,reset,save",
                x_axis_label="Recall",
                y_axis_label="Precision")

    exact_source = ColumnDataSource(df_filtered[df_filtered['is_exact']])
    approx_source = ColumnDataSource(df_filtered[~df_filtered['is_exact']])

    approx_scatter = p2.scatter('recall', 'precision', source=approx_source,
                                size=10, alpha=0.7,
                                line_color="black", line_width=0.5,
                                color=linear_cmap('f1_score', viridis(256),
                                                  df_filtered['f1_score'].min(
                                ),
                                    df_filtered['f1_score'].max()),
                                legend_label='Approximate')

    exact_scatter = p2.scatter('recall', 'precision', source=exact_source,
                               size=14, alpha=0.9,
                               marker='diamond',
                               line_color="black", line_width=1.0,
                               color='#FF4500',
                               legend_label='Exact')

    hover = HoverTool(tooltips=rule_tooltips)
    hover.attachment = "vertical"
    hover.point_policy = "follow_mouse"

    p2.add_tools(hover)

    p2.add_tools(CrosshairTool(line_alpha=0.3))

    p2.legend.location = "top_right"
    p2.legend.click_policy = "hide"

    return p2


def create_roc_plot():
    p3 = figure(title="ROC Space Visualization",
                height=400, width=650,
                tools="pan,box_zoom,wheel_zoom,reset,save",
                x_axis_label="False Positive Rate (1 - Specificity)",
                y_axis_label="True Positive Rate (Recall)",
                x_range=(-0.02, 1.02),
                y_range=(-0.02, 1.02))

    exact_source = ColumnDataSource(df_filtered[df_filtered['is_exact']])
    approx_source = ColumnDataSource(df_filtered[~df_filtered['is_exact']])

    approx_scatter = p3.scatter('false_positive_rate', 'true_positive_rate', source=approx_source,
                                size=10, alpha=0.7,
                                line_color="black", line_width=0.5,
                                color=linear_cmap('f1_score', viridis(256),
                                                  df_filtered['f1_score'].min(
                                ),
                                    df_filtered['f1_score'].max()),
                                legend_label='Approximate')

    exact_scatter = p3.scatter('false_positive_rate', 'true_positive_rate', source=exact_source,
                               size=14, alpha=0.9,
                               marker='diamond',
                               line_color="black", line_width=1.0,
                               color='#FF4500',
                               legend_label='Exact')

    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    p3.line(x_values, y_values, line_dash='dashed',
            color='black', alpha=0.5, legend_label='Random')
    p3.legend.location = 'bottom_right'
    p3.legend.click_policy = "hide"

    hover = HoverTool(tooltips=rule_tooltips)
    hover.attachment = "vertical"
    hover.point_policy = "follow_mouse"

    p3.add_tools(hover)
    p3.add_tools(CrosshairTool(line_alpha=0.3))

    return p3


def create_f1_complexity_plot():
    p4 = figure(title="F1 Score vs Rule Complexity",
                height=400, width=650,
                tools="pan,box_zoom,wheel_zoom,reset,save",
                x_axis_label="Average Rule Complexity (avg_k)",
                y_axis_label="F1 Score")

    exact_source = ColumnDataSource(df_filtered[df_filtered['is_exact']])
    approx_source = ColumnDataSource(df_filtered[~df_filtered['is_exact']])

    approx_scatter = p4.scatter('avg_k', 'f1_score', source=approx_source,
                                size=10, alpha=0.7,
                                line_color="black", line_width=0.5,
                                color=linear_cmap('avg_z', plasma(256),
                                                  df_filtered['avg_z'].min(),
                                                  df_filtered['avg_z'].max()),
                                legend_label='Approximate')

    exact_scatter = p4.scatter('avg_k', 'f1_score', source=exact_source,
                               size=14, alpha=0.9,
                               marker='diamond',
                               line_color="black", line_width=1.0,
                               color='#FF4500',
                               legend_label='Exact')

    mapper = LinearColorMapper(palette=plasma(256),
                               low=df_filtered['avg_z'].min(),
                               high=df_filtered['avg_z'].max())

    color_bar = ColorBar(color_mapper=mapper,
                         title="Avg Z",
                         location=(0, 0),
                         ticker=BasicTicker())
    p4.add_layout(color_bar, 'right')

    hover = HoverTool(tooltips=rule_tooltips)
    hover.attachment = "vertical"
    hover.point_policy = "follow_mouse"

    p4.add_tools(hover)
    p4.legend.location = "top_left"
    p4.legend.click_policy = "hide"

    return p4


def create_feature_plot():
    if top_features:
        features, counts = zip(*top_features)
        features_df = pd.DataFrame({'Feature': features, 'Count': counts})
        feature_source = ColumnDataSource(features_df)

        p5 = figure(title="Most Commonly Used Features in Rules",
                    height=400, width=650,
                    tools="pan,box_zoom,wheel_zoom,reset,save",
                    x_axis_label="Feature",
                    y_axis_label="Frequency",
                    x_range=features)

        p5.vbar(x='Feature', top='Count', width=0.8, source=feature_source,
                color=linear_cmap('Count', Blues[9][::-1], 0, max(counts)),
                line_color="black", line_width=0.5,
                alpha=0.8)

        p5.xaxis.major_label_orientation = np.pi/4

        p5.add_tools(HoverTool(tooltips=[
            ("Feature", "@Feature"),
            ("Count", "@Count")
        ]))

        return p5
    else:

        p5 = figure(title="Most Commonly Used Features in Rules",
                    height=400, width=650)
        p5.text(0.5, 0.5, ['No feature data available'], text_align='center')
        return p5


def create_metrics_plots():
    metrics = ['accuracy', 'precision', 'recall',
               'f1_score', 'specificity', 'loss_db']
    titles = [m.replace('_', ' ').title() for m in metrics]

    plots = []
    for i, metric in enumerate(metrics):

        p = figure(title=titles[i],
                   height=300, width=420,
                   tools="pan,box_zoom,wheel_zoom,reset,save",
                   x_axis_label="Avg Rule Complexity (avg_k)",
                   y_axis_label=titles[i])

        exact_source = ColumnDataSource(df_filtered[df_filtered['is_exact']])
        approx_source = ColumnDataSource(df_filtered[~df_filtered['is_exact']])

        p.scatter('avg_k', metric, source=approx_source,
                  size=8, alpha=0.7,
                  line_color="black", line_width=0.5,
                  color=linear_cmap('f1_score', viridis(256),
                                    df_filtered['f1_score'].min(),
                                    df_filtered['f1_score'].max()),
                  legend_label='Approximate')

        p.scatter('avg_k', metric, source=exact_source,
                  size=12, alpha=0.9,
                  marker='diamond',
                  line_color="black", line_width=1.0,
                  color='#FF4500',
                  legend_label='Exact')

        hover = HoverTool(tooltips=rule_tooltips)
        hover.attachment = "vertical"
        hover.point_policy = "follow_mouse"

        p.add_tools(hover)
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "8pt"
        plots.append(p)

    grid = gridplot(
        [plots[0:3], plots[3:6]],
        sizing_mode='stretch_width'
    )

    return grid


def create_top_rules_table():

    exact_rules = df_filtered[df_filtered['is_exact']].nlargest(5, 'f1_score')
    approx_rules = df_filtered[~df_filtered['is_exact']].nlargest(
        5, 'f1_score')

    combined_rules = pd.concat([exact_rules, approx_rules])
    top_source = ColumnDataSource(combined_rules)

    columns = [

        TableColumn(field="rule", title="Rule", width=600),
        TableColumn(field="solution_type", title="Type", width=100),
        TableColumn(field="accuracy", title="Accuracy",
                    formatter=NumberFormatter(format="0.000")),
        TableColumn(field="precision", title="Precision",
                    formatter=NumberFormatter(format="0.000")),
        TableColumn(field="recall", title="Recall",
                    formatter=NumberFormatter(format="0.000")),
        TableColumn(field="f1_score", title="F1 Score",
                    formatter=NumberFormatter(format="0.000")),
        TableColumn(field="avg_k", title="Avg K",
                    formatter=NumberFormatter(format="0.00")),
        TableColumn(field="avg_z", title="Avg Z",
                    formatter=NumberFormatter(format="0.00")),
        TableColumn(field="occurrence_count", title="Occurrences",
                    formatter=NumberFormatter(format="0"))
    ]

    data_table = DataTable(source=top_source, columns=columns,
                           width=1500, height=350,
                           index_position=None)

    return data_table


def create_feature_complexity_plot():
    p = figure(title="Feature Complexity (avg_z) Effect on Performance",
               height=400, width=650,
               tools="pan,box_zoom,wheel_zoom,reset,save",
               x_axis_label="Average Feature Complexity (avg_z)",
               y_axis_label="F1 Score")

    exact_source = ColumnDataSource(df_filtered[df_filtered['is_exact']])
    approx_source = ColumnDataSource(df_filtered[~df_filtered['is_exact']])

    approx_scatter = p.scatter('avg_z', 'f1_score', source=approx_source,
                               size=10, alpha=0.7,
                               line_color="black", line_width=0.5,
                               color=linear_cmap('avg_k', viridis(256),
                                                 df_filtered['avg_k'].min(),
                                                 df_filtered['avg_k'].max()),
                               legend_label='Approximate')

    exact_scatter = p.scatter('avg_z', 'f1_score', source=exact_source,
                              size=14, alpha=0.9,
                              marker='diamond',
                              line_color="black", line_width=1.0,
                              color='#FF4500',
                              legend_label='Exact')

    mapper = LinearColorMapper(palette=viridis(256),
                               low=df_filtered['avg_k'].min(),
                               high=df_filtered['avg_k'].max())

    color_bar = ColorBar(color_mapper=mapper,
                         title="Avg K",
                         location=(0, 0),
                         ticker=BasicTicker())
    p.add_layout(color_bar, 'right')

    hover = HoverTool(tooltips=rule_tooltips)
    hover.attachment = "vertical"
    hover.point_policy = "follow_mouse"

    p.add_tools(hover)
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def create_jitter_plots():

    measures = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    titles = [m.replace('_', ' ').title() for m in measures]

    jitter_plots = []

    for i, measure in enumerate(measures):

        jitter_data = df_filtered.copy()

        jitter_data['jitter'] = [
            random.uniform(-0.4, 0.4) for _ in range(len(jitter_data))]

        exact_source = ColumnDataSource(jitter_data[jitter_data['is_exact']])
        approx_source = ColumnDataSource(jitter_data[~jitter_data['is_exact']])

        p = figure(title=f"Distribution of {titles[i]}",
                   height=300, width=650,
                   tools="pan,box_zoom,wheel_zoom,reset,save",
                   x_axis_label=titles[i],
                   y_axis_label="")

        p.scatter(measure, 'jitter', source=approx_source,
                  size=10, alpha=0.7,
                  line_color="black", line_width=0.5,
                  color=linear_cmap('f1_score', viridis(256),
                                    df_filtered['f1_score'].min(),
                                    df_filtered['f1_score'].max()),
                  legend_label='Approximate')

        p.scatter(measure, 'jitter', source=exact_source,
                  size=14, alpha=0.9,
                  marker='diamond',
                  line_color="black", line_width=1.0,
                  color='#FF4500',
                  legend_label='Exact')

        p.yaxis.major_tick_line_color = None
        p.yaxis.minor_tick_line_color = None
        p.yaxis.major_label_text_font_size = '0pt'
        p.y_range = Range1d(-0.5, 0.5)

        hover = HoverTool(tooltips=rule_tooltips)
        hover.attachment = "vertical"
        hover.point_policy = "follow_mouse"
        p.add_tools(hover)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        jitter_plots.append(p)

    return jitter_plots


def create_rashomon_plots():

    measures = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    titles = [m.replace('_', ' ').title() for m in measures]

    rashomon_plots = []

    for i, measure in enumerate(measures):

        best_performance = df_filtered[measure].max()

        theta_values = np.linspace(0, 0.2, 100)

        rashomon_sizes = []
        threshold_values = []
        exact_sizes = []

        for theta in theta_values:

            threshold = best_performance - theta

            threshold_values.append(threshold)
            rashomon_size = len(df_filtered[df_filtered[measure] >= threshold])
            rashomon_sizes.append(rashomon_size)

            if 'is_exact' in df_filtered.columns:
                exact_size = len(df_filtered[(df_filtered[measure] >= threshold) &
                                             df_filtered['is_exact']])
                exact_sizes.append(exact_size)
            else:
                exact_sizes.append(0)

        source = ColumnDataSource(data=dict(
            theta=theta_values,
            size=rashomon_sizes,
            exact_size=exact_sizes,
            threshold=threshold_values
        ))

        p = figure(title=f"Rashomon Set Size for {titles[i]}",
                   height=300, width=650,
                   tools="pan,box_zoom,wheel_zoom,reset,save",
                   x_axis_label="θ",
                   y_axis_label="Rashomon Set Size |R(θ)|")

        p.line('theta', 'size', source=source,
               line_width=3, line_color="navy", legend_label="All Solutions")

        p.scatter(x='theta', y='size', source=source,
                  size=5, fill_color="white", line_color="navy")

        p.line('theta', 'exact_size', source=source,
               line_width=3, line_color="#FF4500", line_dash='dashed',
               legend_label="Exact Solutions Only")

        p.scatter(x='theta', y='exact_size', source=source,
                  size=5, fill_color="white", line_color="#FF4500")

        p.add_tools(HoverTool(
            tooltips=[
                ("Theta", "@theta{0.000}"),
                ("Total Set Size", "@size{0}"),
                ("Exact Solutions", "@exact_size{0}"),

                ("Performance Threshold", "@threshold{0.000}")
            ],
            mode="vline"
        ))

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        rashomon_plots.append(p)

    return rashomon_plots


header = Div(text=f"""
<div style="background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
    <h1 style="margin: 0 0 10px 0;">Rule Visualization Dashboard</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", width=1500, height=100)


exact_count = df_filtered['is_exact'].sum(
) if 'is_exact' in df_filtered.columns else 0
approx_count = len(df_filtered) - exact_count

info = Div(text=f"""
<div style="background-color: #ebf5fb; border-left: 4px solid #3498db; padding: 10px 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0;">
    <p>Data source: <strong>{args.fn}</strong></p>
    <p>Total rules: <strong>{len(df_metrics)}</strong></p>
    <p>Rules with recall ≥ {args.min_recall}: <strong>{len(df_filtered)}</strong></p>
    <p>Exact solutions: <strong>{exact_count}</strong> | Approximate solutions: <strong>{approx_count}</strong></p>
</div>
""", width=1500, height=100)


legend_info = Div(text=f"""
<div style="background-color: #fef9e7; border-left: 4px solid #f1c40f; padding: 10px 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0;">
    <p><span style="color: #FF4500; font-size: 16px;">◆</span> <strong>Exact Solutions:</strong> These are mathematically optimal rule lists found using exact solving methods.</p>
    <p><span style="color: #3498DB; font-size: 16px;">●</span> <strong>Approximate Solutions:</strong> These are rule lists found using sampling.</p>
    <p><em>Click on legend items to hide/show different solution types in plots.</em></p>
</div>
""", width=1500, height=100)


plot1 = create_rule_length_plot()
plot2 = create_precision_recall_plot()
plot3 = create_roc_plot()
plot4 = create_f1_complexity_plot()
plot5 = create_feature_plot()
plot6 = create_feature_complexity_plot()
metrics_grid = create_metrics_plots()
top_table = create_top_rules_table()


complexity_header = Div(
    text="<h2>Rule and Feature Complexity Analysis</h2>", width=1500)
performance_header = Div(text="<h2>Performance Analysis</h2>", width=1500)
metrics_header = Div(text="<h2>Metrics vs Rule Complexity</h2>", width=1500)
table_header = Div(text="<h2>Top Rules by F1 Score</h2>", width=1500)

jitter_plots = create_jitter_plots()
rashomon_plots = create_rashomon_plots()


jitter_header = Div(text="<h2>Performance Distribution</h2>", width=1500)
rashomon_header = Div(text="<h2>Rashomon Set Analysis</h2>", width=1500)


dashboard = column(
    header,
    info,
    legend_info,

    complexity_header,
    row(plot1, plot6, sizing_mode='stretch_width', height=400, width=1500),
    row(plot4, plot5, sizing_mode='stretch_width', height=400, width=1500),

    performance_header,
    row(plot2, plot3, sizing_mode='stretch_width', height=400, width=1500),

    jitter_header,
    column(*[row(jitter_plots[i], rashomon_plots[i],
                 sizing_mode='stretch_width', height=300, width=1500)
           for i in range(len(jitter_plots))],
           sizing_mode='stretch_width', width=1500),

    metrics_header,
    column(metrics_grid, sizing_mode='stretch_width', width=1500),

    table_header,
    column(top_table, sizing_mode='stretch_width', width=1500),
    sizing_mode='stretch_width', width=1500
)


save(dashboard)

print(
    f"\nDashboard complete! Open {os.path.abspath(args.o)} to view the visualization")
