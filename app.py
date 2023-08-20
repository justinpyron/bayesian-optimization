import streamlit as st
import numpy as np
from bayesian_optimization import BayesianOptimizer
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# TOY_FUNCTION = lambda x: 20*np.sin(2*x[0]) - 3*x[0]**2 + x[0]
TOY_FUNCTION = lambda x: (20*np.sin(2*x) - 3*x**2 + x)[0]
BOUNDS = [(-5,5)]
KERNEL_BANDWIDTH = 1


st.set_page_config(
    page_title='Bayesian Opt',
    page_icon='🥇',
    layout='wide',
)



def make_plot(optimizer: BayesianOptimizer) -> go.Figure:

    # TODO: add argument for which step to display

    domain = np.sort(optimizer.domain_samples, axis=0)
    posterior = [optimizer.compute_posterior(z) for z in domain]
    mu_list = np.array([p[0] for p in posterior])
    var_list = np.array([p[1] for p in posterior])
    l = mu_list - 1.96*np.sqrt(var_list)
    u = mu_list + 1.96*np.sqrt(var_list)
    EI_list = np.array([optimizer.compute_expected_improvement(z) for z in domain])

    # Make figure
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[2,1],
    )
    fig.update_layout(height=600, width=1200)
    fig.update_layout(legend=dict(
        orientation='h',
        xanchor='center',
        x=0.5,
    ))

    # Bottom chart
    fig.append_trace(go.Scatter(
        x=domain[:,0],
        y=EI_list,
        name='Expected Improvement',
        line=dict(
            color='darkgreen',
            width=3,
        ),
        hoverlabel=dict(namelength=-1),
    ), row=2, col=1)


    # Top chart
    fig.append_trace(go.Scatter(
        x=domain[:,0],
        y=l,
        line=dict(
            color='dodgerblue',
            width=1,
        ),
        name='Estimated Function Uncertainty Lower Bound',
        showlegend=False,
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=domain[:,0],
        y=u,
        fill='tonexty',
        fillcolor='rgba(30,144,255,0.2)', # dodgerblue with less opacity
        line=dict(
            color='dodgerblue',
            width=1,
        ),
        name='Estimated Function Uncertainty Upper Bound',
        showlegend=False,
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=domain[:,0],
        y=mu_list,
        name='Estimated Function',
        line=dict(
            color='dodgerblue',
            width=3,
        ),
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=domain[:,0],
        y=np.array([optimizer.function(z) for z in domain]),
        name='True Function',
        line=dict(
            color='Orange',
            width=3,
        ),
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=optimizer.samples[:,0],
        y=optimizer.sample_values,
        mode='markers',
        name='Oberved Values',
        marker=dict(
            color='Orange',
            size=15,
            line=dict(
                color='DarkOrange',
                width=2
            )
        ),
        hoverlabel=dict(namelength=-1),
    ), row=1, col=1)


    return fig







def initialize_optimizer() -> None:
    '''Returns initialized BayesianOptimizer object'''
    st.session_state.optimizer = BayesianOptimizer(
        function=TOY_FUNCTION,
        domain_bounds=BOUNDS,
        kernel_bandwidth=KERNEL_BANDWIDTH,
    )


if 'optimizer' not in st.session_state:
    initialize_optimizer()


st.title('Bayesian Optimization 🥇')
st.write('👉 Learn how it works on [GitHub](https://github.com/justinpyron/bayesian-optimization)')

col_reset, col_step, col_slider = st.columns(spec=[1,1,3], gap='medium')

with col_reset:
    button_reset = st.button(
        label='Reset',
        help='Reinitialize the Optimizer',
        on_click=initialize_optimizer,
        use_container_width=True,
    )
with col_step:
    button_step = st.button(
        label='Draw Sample',
        help='Take a sample at the point that maximizes Expected Improvement',
        on_click=st.session_state.optimizer.step,
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
st.write('Sample values:')
st.write(st.session_state.optimizer.sample_values)
st.write(st.session_state.optimizer.samples.shape)
st.write(st.session_state.optimizer.samples)

figure = make_plot(st.session_state.optimizer)
st.plotly_chart(figure, use_container_width=True)



