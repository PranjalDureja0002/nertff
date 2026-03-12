"""Data Visualizer component for the Agent Builder.

Takes structured query results and generates professional charts/visualizations
embedded as base64 images in markdown for rendering in the orchestrator chat.
"""

import base64
import io
import json
from typing import Any

from agentcore.custom.custom_node.node import Node
from agentcore.inputs.inputs import (
    DropdownInput,
    HandleInput,
    MessageTextInput,
    BoolInput,
)
from agentcore.schema.data import Data
from agentcore.schema.message import Message
from agentcore.template.field.base import Output
from agentcore.logging import logger


# ---------- Chart Style Presets ----------

STYLE_PRESETS = {
    "corporate": {
        "colors": ["#1e3a5f", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#3b1f2b", "#44bba4", "#e94f37"],
        "bg_color": "#ffffff",
        "text_color": "#333333",
        "grid_color": "#e0e0e0",
        "font_family": "sans-serif",
        "title_size": 14,
        "label_size": 11,
    },
    "modern": {
        "colors": ["#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316", "#eab308", "#22c55e", "#06b6d4"],
        "bg_color": "#1a1a2e",
        "text_color": "#e0e0e0",
        "grid_color": "#2d2d4a",
        "font_family": "sans-serif",
        "title_size": 14,
        "label_size": 11,
    },
    "colorful": {
        "colors": ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#dda0dd", "#98d8c8", "#f7dc6f"],
        "bg_color": "#ffffff",
        "text_color": "#2c3e50",
        "grid_color": "#ecf0f1",
        "font_family": "sans-serif",
        "title_size": 14,
        "label_size": 11,
    },
}


def _chart_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _detect_best_chart_type(columns: list[str], rows: list[list], user_query: str) -> str:
    """Heuristic chart type detection based on data shape."""
    if not rows or not columns:
        return "table"

    num_cols = len(columns)
    num_rows = len(rows)

    # Check for date/time columns
    date_keywords = {"date", "month", "year", "time", "day", "week", "quarter", "period"}
    has_date_col = any(any(kw in col.lower() for kw in date_keywords) for col in columns)

    # Check for numeric columns
    numeric_col_count = 0
    for col_idx in range(num_cols):
        sample = rows[0][col_idx] if rows else None
        if isinstance(sample, (int, float)):
            numeric_col_count += 1

    query_lower = user_query.lower()

    # Distribution / composition
    if any(kw in query_lower for kw in ["distribution", "breakdown", "composition", "proportion", "share"]):
        if num_rows <= 8:
            return "pie"
        return "bar"

    # Trends over time
    if has_date_col and any(kw in query_lower for kw in ["trend", "over time", "monthly", "weekly", "daily", "growth"]):
        return "line"

    # Comparison
    if any(kw in query_lower for kw in ["compare", "comparison", "top", "bottom", "ranking", "best", "worst"]):
        return "bar_horizontal" if num_rows > 10 else "bar"

    # Correlation
    if any(kw in query_lower for kw in ["correlation", "relationship", "scatter", "vs"]):
        if numeric_col_count >= 2:
            return "scatter"

    # Default: time series → line, categorical → bar
    if has_date_col:
        return "line"
    if num_rows <= 6 and numeric_col_count >= 1:
        return "pie"
    return "bar"


def _generate_chart(chart_type: str, columns: list[str], rows: list[list],
                    style_name: str, title: str,
                    chart_options: dict | None = None) -> str:
    """Generate a chart and return as base64 PNG.

    Args:
        chart_options: Optional dict with keys:
            x_axis_label, y_axis_label, show_value_labels, show_legend, auto_axis_labels
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["corporate"])
    colors = style["colors"]

    # Parse chart options (with safe defaults)
    opts = chart_options or {}
    custom_x_label = opts.get("x_axis_label", "")
    custom_y_label = opts.get("y_axis_label", "")
    show_value_labels = opts.get("show_value_labels", True)
    show_legend = opts.get("show_legend", True)
    auto_axis_labels = opts.get("auto_axis_labels", True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(style["bg_color"])
    ax.set_facecolor(style["bg_color"])
    ax.tick_params(colors=style["text_color"])
    ax.xaxis.label.set_color(style["text_color"])
    ax.yaxis.label.set_color(style["text_color"])
    ax.title.set_color(style["text_color"])

    for spine in ax.spines.values():
        spine.set_color(style["grid_color"])

    ax.grid(True, alpha=0.3, color=style["grid_color"])

    # Identify label and value columns
    labels = [str(row[0]) for row in rows]
    # Truncate long labels
    labels = [l[:25] + "..." if len(l) > 25 else l for l in labels]

    if chart_type == "bar":
        if len(columns) > 2:
            # Multiple value columns → grouped bar
            num_groups = len(rows)
            num_bars = len(columns) - 1
            bar_width = 0.8 / num_bars
            x_pos = list(range(num_groups))
            for i in range(1, len(columns)):
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[i]) if row[i] is not None else 0)
                    except (ValueError, TypeError):
                        vals.append(0)
                offset = (i - 1 - (num_bars - 1) / 2) * bar_width
                positions = [x + offset for x in x_pos]
                ax.bar(positions, vals, bar_width * 0.9, label=columns[i],
                       color=colors[(i - 1) % len(colors)], edgecolor="none")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["label_size"])
            if show_legend:
                ax.legend(fontsize=9, facecolor=style["bg_color"], edgecolor=style["grid_color"],
                         labelcolor=style["text_color"])
        else:
            vals = []
            for row in rows:
                try:
                    vals.append(float(row[1]) if len(row) > 1 and row[1] is not None else 0)
                except (ValueError, TypeError):
                    vals.append(0)
            bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
            bars = ax.bar(labels, vals, color=bar_colors, edgecolor="none", width=0.7)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["label_size"])
            # Add value labels on bars
            if show_value_labels:
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:,.0f}" if val > 1 else f"{val:.2f}",
                            ha="center", va="bottom", fontsize=8, color=style["text_color"])

    elif chart_type == "bar_horizontal":
        vals = []
        for row in rows:
            try:
                vals.append(float(row[1]) if len(row) > 1 and row[1] is not None else 0)
            except (ValueError, TypeError):
                vals.append(0)
        bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
        ax.barh(labels, vals, color=bar_colors, edgecolor="none", height=0.7)
        ax.invert_yaxis()

    elif chart_type == "line":
        if len(columns) > 2:
            for i in range(1, len(columns)):
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[i]) if row[i] is not None else 0)
                    except (ValueError, TypeError):
                        vals.append(0)
                ax.plot(labels, vals, marker="o", markersize=5, linewidth=2,
                        color=colors[(i - 1) % len(colors)], label=columns[i])
            if show_legend:
                ax.legend(fontsize=9, facecolor=style["bg_color"], edgecolor=style["grid_color"],
                         labelcolor=style["text_color"])
        else:
            vals = []
            for row in rows:
                try:
                    vals.append(float(row[1]) if len(row) > 1 and row[1] is not None else 0)
                except (ValueError, TypeError):
                    vals.append(0)
            ax.plot(labels, vals, marker="o", markersize=6, linewidth=2.5,
                    color=colors[0])
            ax.fill_between(range(len(vals)), vals, alpha=0.15, color=colors[0])
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["label_size"])

    elif chart_type == "pie":
        vals = []
        for row in rows:
            try:
                vals.append(float(row[1]) if len(row) > 1 and row[1] is not None else 0)
            except (ValueError, TypeError):
                vals.append(0)
        # Filter out zero/negative values
        filtered = [(l, v) for l, v in zip(labels, vals) if v > 0]
        if filtered:
            pie_labels, pie_vals = zip(*filtered)
            pie_colors = [colors[i % len(colors)] for i in range(len(pie_vals))]
            wedges, texts, autotexts = ax.pie(
                pie_vals, labels=pie_labels, colors=pie_colors,
                autopct="%1.1f%%", startangle=140,
                textprops={"fontsize": style["label_size"], "color": style["text_color"]},
            )
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color("white")
                autotext.set_fontweight("bold")
            ax.axis("equal")

    elif chart_type == "scatter":
        if len(columns) >= 3:
            x_vals = []
            y_vals = []
            for row in rows:
                try:
                    x_vals.append(float(row[1]) if row[1] is not None else 0)
                    y_vals.append(float(row[2]) if row[2] is not None else 0)
                except (ValueError, TypeError):
                    x_vals.append(0)
                    y_vals.append(0)
            ax.scatter(x_vals, y_vals, c=colors[0], alpha=0.7, edgecolors=colors[1], s=60)
            # Scatter axis labels: custom > auto > column name
            if custom_x_label:
                ax.set_xlabel(custom_x_label, fontsize=style["label_size"])
            elif auto_axis_labels:
                ax.set_xlabel(columns[1].replace("_", " ").title(), fontsize=style["label_size"])
            if custom_y_label:
                ax.set_ylabel(custom_y_label, fontsize=style["label_size"])
            elif auto_axis_labels:
                ax.set_ylabel(columns[2].replace("_", " ").title(), fontsize=style["label_size"])
        else:
            x_vals = list(range(len(rows)))
            y_vals = []
            for row in rows:
                try:
                    y_vals.append(float(row[1]) if len(row) > 1 and row[1] is not None else 0)
                except (ValueError, TypeError):
                    y_vals.append(0)
            ax.scatter(x_vals, y_vals, c=colors[0], alpha=0.7, s=60)

    else:
        # Fallback: just show text
        ax.text(0.5, 0.5, "Chart type not supported", ha="center", va="center",
                fontsize=14, color=style["text_color"], transform=ax.transAxes)

    # --- Apply axis labels (all chart types except pie and scatter which handles its own) ---
    if chart_type not in ("pie", "scatter"):
        # X-axis label
        x_label = custom_x_label
        if not x_label and auto_axis_labels and columns:
            x_label = columns[0].replace("_", " ").title()
        if x_label:
            ax.set_xlabel(x_label, fontsize=style["label_size"], color=style["text_color"])

        # Y-axis label
        y_label = custom_y_label
        if not y_label and auto_axis_labels and len(columns) > 1:
            if len(columns) == 2:
                y_label = columns[1].replace("_", " ").title()
            else:
                y_label = "Value"
        if y_label:
            ax.set_ylabel(y_label, fontsize=style["label_size"], color=style["text_color"])

    if chart_type != "pie":
        ax.set_title(title, fontsize=style["title_size"], fontweight="bold",
                     color=style["text_color"], pad=15)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"
        ))
    else:
        ax.set_title(title, fontsize=style["title_size"], fontweight="bold",
                     color=style["text_color"], pad=20)

    plt.tight_layout()
    b64 = _chart_to_base64(fig)
    plt.close(fig)
    return b64


class DataVisualizerComponent(Node):
    """Generates professional charts and visualizations from query data.

    Takes structured data (from NL-to-SQL or other sources) and creates
    beautiful charts rendered as embedded images in markdown.
    """

    display_name = "Data Visualizer"
    description = "Generate charts and visualizations from query results."
    icon = "bar-chart-3"
    name = "DataVisualizer"
    hidden = True

    inputs = [
        HandleInput(
            name="query_data",
            display_name="Query Data",
            input_types=["Data"],
            required=True,
            info="Structured data from NL-to-SQL (use the 'Raw Data' output).",
        ),
        MessageTextInput(
            name="user_query",
            display_name="Original Question",
            value="",
            info="The original user question (helps determine the best chart type).",
        ),
        BoolInput(
            name="show_sql",
            display_name="Show Generated SQL",
            value=True,
            info="Include the generated SQL query in the output.",
        ),
        DropdownInput(
            name="chart_type",
            display_name="Chart Type",
            options=["auto", "bar", "bar_horizontal", "line", "pie", "scatter", "table"],
            value="auto",
            info="Chart type to generate. 'auto' detects the best type based on data shape.",
        ),
        DropdownInput(
            name="chart_style",
            display_name="Chart Style",
            options=["corporate", "modern", "colorful"],
            value="corporate",
            info="Visual style preset for the chart.",
        ),
        BoolInput(
            name="include_data_table",
            display_name="Include Data Table",
            value=True,
            info="Also include the data as a markdown table below the chart.",
        ),
    ]

    outputs = [
        Output(
            display_name="Visualization",
            name="visualization",
            method="generate_visualization",
            types=["Message"],
        ),
    ]

    def _build_query_summary(self, data_dict: dict, columns: list, rows: list) -> str:
        """Build the NL-to-SQL results summary from raw data fields."""
        parts = []

        row_count = data_dict.get("row_count", len(rows))
        exec_time = data_dict.get("execution_time_ms", "")
        generated_sql = data_dict.get("generated_sql", "")

        meta = [f"{row_count} rows"]
        if exec_time:
            meta.append(f"{exec_time}ms")
        parts.append(f"**Query Results** ({', '.join(meta)})\n")

        if self.show_sql and generated_sql:
            parts.append(f"**Generated SQL:**\n```sql\n{generated_sql}\n```\n")

        if self.include_data_table and len(rows) <= 50:
            parts.append(_format_results_table(columns, rows))

        return "\n".join(parts)

    def generate_visualization(self) -> Message:
        """Generate chart visualization from query data."""
        # Extract data
        raw_data = self.query_data
        if isinstance(raw_data, Data):
            data_dict = raw_data.data
        elif isinstance(raw_data, dict):
            data_dict = raw_data
        else:
            self.status = "Invalid input data"
            return Message(text="Error: Expected structured Data input with 'columns' and 'rows'.")

        columns = data_dict.get("columns", [])
        rows = data_dict.get("rows", [])
        user_query = self.user_query or data_dict.get("user_query", "Data Visualization")

        if not columns or not rows:
            self.status = "No data to visualize"
            if data_dict.get("error"):
                return Message(text=data_dict.get("message", "Error in upstream data."))
            return Message(text="No data available to visualize.")

        # Determine chart type
        chart_type = self.chart_type
        if chart_type == "auto":
            chart_type = _detect_best_chart_type(columns, rows, user_query)
            logger.info(f"Auto-detected chart type: {chart_type}")

        # Build query summary (SQL + data table)
        query_summary = self._build_query_summary(data_dict, columns, rows)

        if chart_type == "table":
            # Just return the summary (already has the data table)
            self.status = f"Table: {len(rows)} rows"
            return Message(text=query_summary)

        # Generate chart
        try:
            title = user_query[:80] + ("..." if len(user_query) > 80 else "")
            chart_options = {
                "x_axis_label": "",
                "y_axis_label": "",
                "show_value_labels": True,
                "show_legend": True,
                "auto_axis_labels": True,
            }
            b64_image = _generate_chart(chart_type, columns, rows, self.chart_style, title, chart_options)

            parts = []
            # Query summary first (SQL + data table)
            parts.append(query_summary)
            parts.append("\n---\n")
            # Then the chart
            parts.append(f"![{title}](data:image/png;base64,{b64_image})\n")

            self.status = f"Chart: {chart_type} ({len(rows)} rows)"
            return Message(text="\n".join(parts))

        except Exception as e:
            logger.error(f"Chart generation failed: {e!s}")
            self.status = f"Fallback table (chart error)"
            return Message(text=query_summary + "\n\n_Chart generation failed._")


def _format_results_table(columns: list[str], rows: list[list], max_display: int = 50) -> str:
    """Format results as markdown table."""
    if not rows:
        return "_No data._"

    display_rows = rows[:max_display]
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    row_lines = []
    for row in display_rows:
        cells = []
        for val in row:
            if val is None:
                cells.append("NULL")
            elif isinstance(val, float):
                cells.append(f"{val:,.2f}")
            elif isinstance(val, int) and abs(val) > 999:
                cells.append(f"{val:,}")
            else:
                cells.append(str(val)[:80])
        row_lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join([header, separator] + row_lines)
    if len(rows) > max_display:
        table += f"\n\n_Showing {max_display} of {len(rows)} rows._"
    return table