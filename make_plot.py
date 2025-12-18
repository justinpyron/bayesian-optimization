import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bayesian_optimization import BayesianOptimizer


def make_plot(
    optimizer: BayesianOptimizer,
    step_to_display: int,
) -> go.Figure:
    domain = np.sort(optimizer.domain_samples, axis=0)
    true_function = np.array([optimizer.function(z) for z in domain])
    expected_improvement = np.array(
        [optimizer.compute_expected_improvement(z, step_to_display) for z in domain]
    )
    posterior = [optimizer.compute_posterior(z, step_to_display) for z in domain]
    posterior_mean = np.array([_mean for (_mean, _var) in posterior])
    posterior_var = np.array([_var for (_mean, _var) in posterior])
    lower_bound = posterior_mean - 1.96 * np.sqrt(posterior_var)
    upper_bound = posterior_mean + 1.96 * np.sqrt(posterior_var)

    # Make figure
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[3, 1],
        shared_xaxes=True,
        vertical_spacing=0,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
        ),
        xaxis_range=[
            optimizer.domain_bounds[0][0],
            optimizer.domain_bounds[0][1],
        ],
    )
    fig.update_yaxes(
        range=[
            true_function.min() - 0.5 * (true_function.max() - true_function.min()),
            true_function.max() + 0.5 * (true_function.max() - true_function.min()),
        ],
        row=1,
        col=1,
    )

    # Bottom chart
    fig.append_trace(
        go.Scatter(
            x=domain[:, 0],
            y=expected_improvement,
            name="Expected Improvement",
            line=dict(
                color="springgreen",
                width=3,
            ),
            hoverlabel=dict(namelength=-1),
        ),
        row=2,
        col=1,
    )

    # Top chart
    fig.append_trace(
        go.Scatter(
            name="Estimated Function Uncertainty Lower Bound",
            x=domain[:, 0],
            y=lower_bound,
            line=dict(
                color="dodgerblue",
                width=0.5,
            ),
            hoverlabel=dict(namelength=-1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Estimated Function Uncertainty Upper Bound",
            x=domain[:, 0],
            y=upper_bound,
            line=dict(
                color="dodgerblue",
                width=0.5,
            ),
            hoverlabel=dict(namelength=-1),
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(30,144,255,0.2)",  # dodgerblue with less opacity
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Estimated Function",
            x=domain[:, 0],
            y=posterior_mean,
            line=dict(
                color="dodgerblue",
                width=3,
            ),
            hoverlabel=dict(namelength=-1),
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="True Function",
            x=domain[:, 0],
            y=true_function,
            line=dict(
                color="Orange",
                width=5,
            ),
            hoverlabel=dict(namelength=-1),
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            name="Oberved Values (label = order of observation)",
            x=optimizer.sample_points[:, 0][:step_to_display],
            y=optimizer.sample_values[:step_to_display],
            text=np.arange(1, len(optimizer.sample_values) + 1).astype(str),
            mode="markers+text",
            marker=dict(
                color="Orange",
                size=18,
                line=dict(
                    color="DarkOrange",
                    # color='lightslategray',
                    width=1.5,
                ),
            ),
            hoverlabel=dict(namelength=-1),
        ),
        row=1,
        col=1,
    )

    fig.add_vline(
        x=domain[np.argmax(expected_improvement)][0],
        line_width=3,
        line_dash="dash",
        line_color="springgreen",
    )

    return fig
