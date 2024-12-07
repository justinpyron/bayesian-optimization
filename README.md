# bayesian-optimization
A Bayesian Optimization solver.

Bayesian Optimization is a _derivative-free_ method for optimizing _black-box_ functions.
- _Derivative-free_. This means derivatives/gradients of the function are not needed. This is in stark contrast to most optimization methods which depend entirely on derivatives.
- _Black-box_. This means no knowledge of the functional form is needed. Only the ability to evaluate the function is required.

Together, these two properties make Bayesian Optimization a powerful tool for optimizing complex functions that are expensive to evaluate.

# Project Organization
```
├── README.md                 <- Overview
├── app.py                    <- Streamlit web app frontend
├── bayesian_optimization.py  <- Bayesian Optimization solver class
├── make_plot.py              <- Utils for generating plotly charts to visualize the algorithm
├── pyproject.toml            <- Poetry config specifying Python environment dependencies
├── poetry.lock               <- Locked dependencies to ensure consistent installs
```

# Installation
This project uses [Poetry](https://python-poetry.org/docs/) to manage its Python environment.

1. Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```
poetry install
```

# Usage
A Streamlit web app is the frontend for interacting with the solver.

The app can be accessed at https://bayesian-optimization.streamlit.app.

Alternatively, the app can be run locally with
```
poetry run streamlit run app.py
```
