"""Data Visualizer Tool — LCToolNode wrapper for Worker Node integration.

Generates charts from structured data.  This tool does NOT query the database
itself — it expects data that was already fetched by the talk_to_data tool.
The Worker Node agent calls talk_to_data first, then passes the results here
when the user wants a chart, graph, or visualization.

Reuses chart generation helpers from the existing data_visualizer module.
"""

import json
from typing import List

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from agentcore.base.langchain_utilities.model import LCToolNode
from agentcore.field_typing import Tool
from agentcore.inputs.inputs import BoolInput, DropdownInput, MessageTextInput
from agentcore.schema.data import Data
from agentcore.logging import logger

# Reuse helpers from existing data_visualizer component
from agentcore.components.tools.data_visualizer import (
    _detect_best_chart_type,
    _generate_chart,
    _format_results_table,
)


class DataVisualizerTool(LCToolNode):
    """Generate charts from query data — as a Worker Node tool.

    Accepts structured data (columns + rows as JSON) and produces a
    professional chart embedded as a base64 image in markdown.
    Does NOT connect to a database or LLM — it only does visualization.
    The Worker Node agent should call talk_to_data first to get the data,
    then call this tool to generate a chart from that data.
    """

    display_name = "Data Visualizer Tool"
    description = (
        "A tool that generates charts and visualizations from query result data. "
        "Wire this into a Worker Node's Tools input alongside the Talk to Data Tool."
    )
    icon = "bar-chart-3"
    name = "DataVisualizerTool"

    inputs = [
        DropdownInput(
            name="chart_style",
            display_name="Chart Style",
            options=["corporate", "modern", "colorful"],
            value="corporate",
            info="Visual style preset for generated charts.",
            advanced=True,
        ),
        MessageTextInput(
            name="x_axis_label",
            display_name="X-Axis Label",
            value="",
            info="Custom label for the X-axis. Leave empty to auto-detect from column names.",
            advanced=True,
        ),
        MessageTextInput(
            name="y_axis_label",
            display_name="Y-Axis Label",
            value="",
            info="Custom label for the Y-axis. Leave empty to auto-detect from column names.",
            advanced=True,
        ),
        BoolInput(
            name="show_value_labels",
            display_name="Show Value Labels",
            value=True,
            info="Display data values directly on chart elements (bars, points).",
            advanced=True,
        ),
        BoolInput(
            name="show_legend",
            display_name="Show Legend",
            value=True,
            info="Display the chart legend (applies to multi-series charts).",
            advanced=True,
        ),
        BoolInput(
            name="auto_axis_labels",
            display_name="Auto Axis Labels from Columns",
            value=True,
            info="Automatically set axis labels from column names when no custom labels are provided.",
            advanced=True,
        ),
    ]

    # ----- Pydantic schema for the tool arguments -----

    class _ToolSchema(BaseModel):
        data_json: str = Field(
            ...,
            description=(
                "The query result data as a JSON string. "
                "Use the exact JSON from the <data_json> block returned by the talk_to_data tool. "
                "Format: {\"columns\": [\"col1\", \"col2\"], \"rows\": [[\"val1\", 123], [\"val2\", 456]]}"
            ),
        )
        chart_type: str = Field(
            default="auto",
            description=(
                "The type of chart to generate. Options: auto, bar, line, pie, scatter. "
                "'auto' picks the best chart based on the data shape. Use 'pie' for distributions, "
                "'line' for trends over time, 'bar' for comparisons, 'scatter' for correlations."
            ),
        )
        chart_title: str = Field(
            default="",
            description="Title for the chart. If empty, a title will be inferred from the data.",
        )

    # ----- LCToolNode interface -----

    def run_model(self) -> List[Data]:
        """Standalone execution (when not used as a tool)."""
        return [Data(data={"message": "Use via Worker Node tool — pass data_json from talk_to_data."})]

    def build_tool(self) -> Tool:
        """Build the LangChain StructuredTool for Worker Node."""
        return StructuredTool.from_function(
            name="data_visualizer",
            description=(
                "Generate a chart or visualization from query result data. "
                "IMPORTANT: Do NOT call this tool directly for a user question. "
                "You MUST first call the talk_to_data tool to get the data, then pass "
                "the <data_json> content from that tool's response into this tool's data_json parameter. "
                "Use this when the user asks for a chart, graph, plot, or visualization."
            ),
            func=self._tool_invoke,
            args_schema=self._ToolSchema,
        )

    # ----- Core logic -----

    def _tool_invoke(self, data_json: str, chart_type: str = "auto", chart_title: str = "") -> str:
        """Entry point when called by the Worker Node agent."""
        try:
            return self._generate_visualization(data_json, chart_type, chart_title)
        except Exception as e:
            raise ToolException(str(e)) from e

    def _generate_visualization(self, data_json: str, chart_type: str, chart_title: str) -> str:
        """Parse the JSON data and generate a chart."""

        # Parse the JSON data
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON data — {e!s}"

        columns = data.get("columns", [])
        rows = data.get("rows", [])

        if not columns or not rows:
            return "Error: No data to visualize. The data_json must contain 'columns' and 'rows'."

        # Auto-detect chart type if needed
        if chart_type == "auto":
            chart_type = _detect_best_chart_type(columns, rows, chart_title)
            logger.info(f"DataVisualizerTool: auto-detected chart type: {chart_type}")

        # Build output
        parts = [f"**Chart** ({len(rows)} data points)\n"]

        # Include data table if small enough
        if len(rows) <= 50:
            parts.append(_format_results_table(columns, rows))

        # Generate chart
        if chart_type != "table":
            try:
                title = chart_title if chart_title else f"Data Visualization ({len(rows)} rows)"
                title = title[:80] + ("..." if len(title) > 80 else "")
                chart_options = {
                    "x_axis_label": self.x_axis_label or "",
                    "y_axis_label": self.y_axis_label or "",
                    "show_value_labels": self.show_value_labels,
                    "show_legend": self.show_legend,
                    "auto_axis_labels": self.auto_axis_labels,
                }
                b64_image = _generate_chart(chart_type, columns, rows, self.chart_style, title, chart_options)
                parts.append("\n---\n")
                parts.append(f"![{title}](data:image/png;base64,{b64_image})\n")
                logger.info(f"DataVisualizerTool: generated {chart_type} chart")
            except Exception as e:
                logger.error(f"Chart generation failed: {e!s}")
                parts.append("\n\n_Chart generation failed — showing data table only._")

        return "\n".join(parts)