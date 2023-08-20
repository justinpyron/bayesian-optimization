import streamlit as st
import numpy as np
from bayesian_optimization import BayesianOptimizer
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Callable


FUNCTION_ZOO = {
    'Function 1': {
        'function': lambda x: (20*np.sin(2*x) - 3*x**2 + x)[0],
        'domain_bounds': [(-5,5)],
        'kernel_bandwidth': 1,
    },
    'Function 2': {
        'function': lambda x: (- 1e-5 * (x + 80) * (x + 30) * (x - 20) * (x - 95) + 1000)[0],
        'domain_bounds': [(-100,100)],
        'kernel_bandwidth': 20,
    }
}


st.set_page_config(
    page_title='Bayesian Opt',
    page_icon='🥇',
    layout='wide',
)
st.write(
    '<style>div.block-container{padding-top:1rem;}</style>',
    unsafe_allow_html=True,
) # Reduce whitespace above title 


def make_plot(
    optimizer: BayesianOptimizer,
    step_to_display: int,
) -> go.Figure:
    domain = np.sort(optimizer.domain_samples, axis=0)
    true_function = np.array([optimizer.function(z) for z in domain])
    expected_improvement = np.array([
        optimizer.compute_expected_improvement(z, step_to_display) for z in domain
    ])
    posterior = [optimizer.compute_posterior(z, step_to_display) for z in domain]
    posterior_mean = np.array([_mean for (_mean, _var) in posterior])
    posterior_var = np.array([_var for (_mean, _var) in posterior])
    lower_bound = posterior_mean - 1.96*np.sqrt(posterior_var)
    upper_bound = posterior_mean + 1.96*np.sqrt(posterior_var)

    # Make figure
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[3,1],
        shared_xaxes=True,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation='h',
            xanchor='center',
            x=0.5,
        ),
        xaxis_range=[
            optimizer.domain_bounds[0][0],
            optimizer.domain_bounds[0][1],
        ]
    )
    fig.update_yaxes(range=[
        true_function.min() - 0.5*(true_function.max() - true_function.min()),
        true_function.max() + 0.5*(true_function.max() - true_function.min()),
    ], row=1, col=1)

    # Bottom chart
    fig.append_trace(go.Scatter(
        x=domain[:,0],
        y=expected_improvement,
        name='Expected Improvement',
        line=dict(
            color='springgreen',
            width=4,
        ),
        hoverlabel=dict(namelength=-1),
    ), row=2, col=1)

    # Top chart
    fig.append_trace(go.Scatter(
        name='Estimated Function Uncertainty Lower Bound',
        x=domain[:,0],
        y=lower_bound,
        line=dict(
            color='dodgerblue',
            width=1,
        ),
        hoverlabel=dict(namelength=-1),
        showlegend=False,
    ), row=1, col=1)
    fig.append_trace(go.Scatter(
        name='Estimated Function Uncertainty Upper Bound',
        x=domain[:,0],
        y=upper_bound,
        line=dict(
            color='dodgerblue',
            width=1,
        ),
        hoverlabel=dict(namelength=-1),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(30,144,255,0.2)', # dodgerblue with less opacity
    ), row=1, col=1)
    fig.append_trace(go.Scatter(
        name='Estimated Function',
        x=domain[:,0],
        y=posterior_mean,
        line=dict(
            color='dodgerblue',
            width=4,
        ),
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)
    fig.append_trace(go.Scatter(
        name='True Function',
        x=domain[:,0],
        y=true_function,
        line=dict(
            color='Orange',
            width=4,
        ),
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)
    fig.append_trace(go.Scatter(
        name='Oberved Values (label = order of observation)',
        x=optimizer.sample_points[:,0][:step_to_display],
        y=optimizer.sample_values[:step_to_display],
        text=np.arange(1,len(optimizer.sample_values)+1).astype(str),
        mode='markers+text',
        marker=dict(
            color='Orange',
            size=18,
            line=dict(
                color='DarkOrange',
                # color='lightslategray',
                width=1.5
            )
        ),
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)

    return fig


def initialize_optimizer(
    function: Callable[[np.array], float],
    domain_bounds: list[tuple[float, float]],
    kernel_bandwidth: float,
) -> None:
    '''Returns initialized BayesianOptimizer object'''
    st.session_state.optimizer = BayesianOptimizer(
        function=function,
        domain_bounds=domain_bounds,
        kernel_bandwidth=kernel_bandwidth,
    )

st.title('Bayesian Optimization 🥇')
st.write('👉 Learn how it works on [GitHub](https://github.com/justinpyron/bayesian-optimization)')
function_selectbox = st.selectbox(
    label='Select a toy function',
    options=[
        'Function 1',
        'Function 2',
    ],
)
function_specs = FUNCTION_ZOO[function_selectbox]

if 'optimizer' not in st.session_state:
    initialize_optimizer(
        function=function_specs['function'],
        domain_bounds=function_specs['domain_bounds'],
        kernel_bandwidth=function_specs['kernel_bandwidth'],
    )

col_bottoms, col_slider = st.columns(spec=[1,2], gap='medium')
with col_bottoms:
    button_reset = st.button(
        label='Reset',
        on_click=initialize_optimizer,
        use_container_width=True,
        kwargs={
            'function': function_specs['function'],
            'domain_bounds': function_specs['domain_bounds'],
            'kernel_bandwidth': function_specs['kernel_bandwidth'],
        }
    )
    button_step = st.button(
        label='Draw Sample',
        on_click=st.session_state.optimizer.take_step,
        type='primary',
        use_container_width=True,
    )
with col_slider:
    slider = st.slider(
        label='Step',
        min_value=1,
        max_value=len(st.session_state.optimizer.sample_values),
        value=len(st.session_state.optimizer.sample_values),
    )

figure = make_plot(st.session_state.optimizer, step_to_display=slider)
st.plotly_chart(figure, use_container_width=True)
