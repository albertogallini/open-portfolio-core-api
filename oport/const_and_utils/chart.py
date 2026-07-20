import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from openbb_charting.core.openbb_figure import OpenBBFigure
from openbb_core.app.model.charts.chart import Chart


def create_chart(
    portfolio_data: dict, benchmark_data: dict, field: str
) -> OpenBBFigure:
    fig = OpenBBFigure(create_backend=True)
    if not benchmark_data:
        fig.add_trace(
            go.Scatter(
                x=portfolio_data["date"],
                y=portfolio_data[field],
                mode="lines",
                name=f"{field} (Portfolio)",
                line=dict(color="white"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=portfolio_data["date"],
                y=portfolio_data[field],
                mode="lines",
                name=f"{field} (Portfolio)",
                line=dict(color="white"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_data["date"],
                y=benchmark_data[field],
                mode="lines",
                name=f"{field} (Bemchmark)",
                line=dict(color="orange"),
            )
        )

    # Extracting title and axis labels from the data
    title = f"{field} over Time"
    x_axis_title = "date"  # Assuming the first column is the date
    y_axis_title = field

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        showlegend=False,  # Hide the legend
    )

    fig.show(renderer="browser")
    return fig


def create_charts(
    portfolio_data: dict, benchmark_data: dict, fields_to_plot: list, title: str
) -> OpenBBFigure:
    # Align data by date if benchmark_data is provided
    if benchmark_data:
        dates = sorted(set(portfolio_data["date"]).intersection(benchmark_data["date"]))
        portfolio_data_aligned = {
            field: [
                portfolio_data[field][portfolio_data["date"].index(date)]
                for date in dates
            ]
            for field in fields_to_plot
        }
        benchmark_data_aligned = {
            field: [
                benchmark_data[field][benchmark_data["date"].index(date)]
                for date in dates
            ]
            for field in fields_to_plot
        }
    else:
        dates = portfolio_data["date"]
        portfolio_data_aligned = {
            field: portfolio_data[field] for field in fields_to_plot
        }

    num_fields = len(fields_to_plot)
    rows = num_fields * 2 if benchmark_data else num_fields

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.043,
        subplot_titles=[f"{field} Portfolio vs Benchmark" for field in fields_to_plot],
    )

    if benchmark_data:
        spread_suffix = " Spread"
        fields_to_plot_pvb = []
        for item in fields_to_plot:
            fields_to_plot_pvb.append(item)
            fields_to_plot_pvb.append(item + spread_suffix)
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.043,
            subplot_titles=[
                f"{field} Portfolio vs Benchmark" for field in fields_to_plot_pvb
            ],
        )

    for i, field in enumerate(fields_to_plot):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_data_aligned[field],
                mode="lines",
                name="Portfolio",
                line=dict(color="orange"),
                showlegend=(i == 0),
            ),
            row=(i * 2) + 1 if benchmark_data else i + 1,
            col=1,
        )

        if benchmark_data:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=benchmark_data_aligned[field],
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="white"),
                    showlegend=(i == 0),
                ),
                row=(i * 2) + 1,
                col=1,
            )

            # Calculate the spread
            spread = [
                p - b
                for p, b in zip(
                    portfolio_data_aligned[field], benchmark_data_aligned[field]
                )
            ]
            spread_positive = [max(0, s) for s in spread]
            spread_negative = [min(0, s) for s in spread]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=spread_positive,
                    fill="tozeroy",
                    mode="none",
                    name="Spread Positive",
                    fillcolor="green",
                    showlegend=False,
                ),
                row=(i * 2) + 2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=spread_negative,
                    fill="tozeroy",
                    mode="none",
                    name="Spread Negative",
                    fillcolor="red",
                    showlegend=False,
                ),
                row=(i * 2) + 2,
                col=1,
            )

            # Add title and axis labels for each subplot
            fig.update_yaxes(
                title_text=f"{field} Value", row=(i * 2) + 1, col=1, showgrid=True
            )
            fig.update_yaxes(title_text="Spread", row=(i * 2) + 2, col=1, showgrid=True)
        else:
            fig.update_yaxes(
                title_text=f"{field} Value", row=i + 1, col=1, showgrid=True
            )

    # Add a shared x-axis title and remove grid lines
    fig.update_xaxes(title_text="Date", row=rows, col=1, showgrid=False, showline=True)

    fig.update_layout(
        title=title,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        height=300 * rows,  # Adjust height based on the number of fields
        showlegend=True,  # Ensure legend is shown
    )

    fig.show(renderer="browser")
    return OpenBBFigure(fig)
