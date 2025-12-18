import numpy as np
import streamlit as st

from bayesian_optimization import BayesianOptimizer
from make_plot import make_plot

FUNCTION_ZOO = {
    "Function 1": {
        "function": lambda x: (20 * np.sin(2 * x) - 3 * x**2 + x)[0],
        "domain_bounds": [(-5, 5)],
        "kernel_bandwidth": 1,
    },
    "Function 2": {
        "function": lambda x: (
            -1e-5 * (x + 80) * (x + 30) * (x - 20) * (x - 95) + 1000
        )[0],
        "domain_bounds": [(-100, 100)],
        "kernel_bandwidth": 30,
    },
}


def initialize_optimizer() -> None:
    """Returns initialized BayesianOptimizer object"""
    function_specs = FUNCTION_ZOO[st.session_state.function_selection]
    st.session_state.optimizer = BayesianOptimizer(
        function=function_specs["function"],
        domain_bounds=function_specs["domain_bounds"],
        kernel_bandwidth=function_specs["kernel_bandwidth"],
    )


st.set_page_config(
    page_title="Bayesian Opt",
    page_icon="🥇",
    layout="wide",
)
st.write(
    "<style>div.block-container{padding-top:1rem;}</style>",
    unsafe_allow_html=True,
)  # Reduce whitespace above title
st.title("Bayesian Optimization 🥇")
st.write(
    "👉 Learn how it works on [GitHub](https://github.com/justinpyron/bayesian-optimization)"
)
if "function_selection" not in st.session_state:
    st.session_state.function_selection = "Function 1"
if "optimizer" not in st.session_state:
    initialize_optimizer()
function_selectbox = st.selectbox(
    label="Select a function to optimize",
    options=[
        "Function 1",
        "Function 2",
    ],
)
if st.session_state.function_selection != function_selectbox:
    st.session_state.function_selection = function_selectbox
    initialize_optimizer()
col_buttons, col_slider = st.columns(spec=[1, 2], gap="medium")
with col_buttons:
    button_reset = st.button(
        label="Reset",
        on_click=initialize_optimizer,
        use_container_width=True,
    )
    button_step = st.button(
        label="Draw Sample",
        on_click=st.session_state.optimizer.take_step,
        type="primary",
        use_container_width=True,
    )
with col_slider:
    if len(st.session_state.optimizer.sample_values) > 1:
        step_to_display = st.slider(
            label="Step",
            min_value=1,
            max_value=len(st.session_state.optimizer.sample_values),
            value=len(st.session_state.optimizer.sample_values),
        )
    else:
        step_to_display = 1
figure = make_plot(st.session_state.optimizer, step_to_display)
st.plotly_chart(figure, use_container_width=True)
