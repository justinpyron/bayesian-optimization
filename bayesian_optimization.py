import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Callable


class BayesianOptimizer:
    '''
    Class for performing Bayesian Optimization.
    See https://arxiv.org/abs/1807.02811 for theory details.
    '''
    def __init__(
        self,
        function: Callable[[np.array], float],
        domain_bounds: list[tuple[float, float]],
        kernel_bandwidth: float,
    ):
        self.function = function
        self.domain_bounds = domain_bounds
        self.kernel_bandwidth = kernel_bandwidth
        self.domain_samples = self.uniformly_sample_domain(500)
        # Initialize optimizer with one sample
        self.sample_points = self.uniformly_sample_domain(number_of_samples=1)
        self.sample_values = np.array([self.function(x) for x in self.sample_points])


    def uniformly_sample_domain(
            self,
            number_of_samples: int,
        ) -> np.array:
        return np.array([
            np.random.uniform(lower, upper, number_of_samples)
            for (lower, upper) in self.domain_bounds
        ]).T


    def compute_sample_variance(self) -> float:
        '''Compute the variance of function values observed thus far'''
        return self.sample_values.var() if len(self.sample_values) > 1 else 1


    def kernel(
        self,
        x: np.array,
        y: np.array,
    ) -> float:
        '''
        Return the covariance between two points.

        Parameter self.kernel_bandwidth controls how high covariance
        is as a function of distance between two points.
        Higher values of self.kernel_bandwidth lead to higher covariance.
        
        Returns the variance of self.function at a 
        single point p when fed that point twice,
        i.e. kernel(p, p).

        Since the true variance of the underlying data
        distribution is unknown, it is estimated with
        the variance of samples that have been collected 
        so far. There isn't a theoretical justification;
        it's simply seemed to me the most data-driven and
        and subjective estimation method.
        '''
        return self.compute_sample_variance() * np.exp(
            -0.5 * np.square(x - y).sum() / (self.kernel_bandwidth ** 2)
        )


    def compute_posterior(
        self,
        sample_point: np.array,
        step: int = None,
    ) -> tuple[np.array, np.array]:
        '''
        Compute the mean and variance of a sample point, conditional on samples seen thus far.

        If [x_1, x_2] ~ N(mean, var),
        where
            mean = [mean_1, mean_2]
            var = [[var_1_1, var_1_2], [var_2_1, var_2_2]]
        then (x_1 | x_2 = observed_values) ~ N(mean_tilde, var_tilde),
        where
            mean_tilde = mean_1 + var_1_2 (var_2_2 ^ -1) (observed_values - mean_2)
            var_tilde  = var_1_1 - var_1_2 (var_2_2 ^ -1) var_2_1
        
        In our setting,
            x_1 = sample_point
            x_2 = observed_values
        '''
        observed_samples = self.sample_points[:step]
        observed_values = self.sample_values[:step]
        mean_1 = observed_values.mean()
        mean_2 = observed_values.mean() * np.ones_like(observed_values)
        var_1_1 = self.kernel(sample_point, sample_point)
        var_1_2 = var_2_1 = np.array([
            self.kernel(sample_point, x) for x in observed_samples
        ])
        var_2_2 = np.array([
            [self.kernel(x, y) for x in observed_samples] for y in observed_samples
        ])
        mean_tilde = mean_1 + var_1_2.dot(
            np.linalg.solve(var_2_2, observed_values - mean_2)
        )
        var_tilde = var_1_1 - var_1_2.dot(
            np.linalg.solve(var_2_2, var_2_1)
        )
        return mean_tilde, var_tilde + 1e-5  # Add small value for numerical stability


    def compute_expected_improvement(
        self,
        point: np.array,
        step: int = None,
    ) -> float:
        '''Compute expected improvement over highest observed point'''
        best_yet = self.sample_values[:step].max()
        mean, variance = self.compute_posterior(point, step)
        st_dev = np.sqrt(variance)
        if st_dev > 1e-5:
            Z = (mean - best_yet) / st_dev
            return (mean - best_yet)*norm.cdf(Z) + st_dev*norm.pdf(Z)
        else:
            return np.maximum(0, mean - best_yet)


    def take_step(self) -> None:
        '''Take a sample at the point that maximizes Expected Improvement'''
        expected_improvement = np.array([
            self.compute_expected_improvement(z) for z in self.domain_samples
        ])
        next_sample = self.domain_samples[np.argmax(expected_improvement)]
        self.sample_points = np.append(
            self.sample_points,
            np.expand_dims(next_sample, axis=0),
            axis=0,
        )
        self.sample_values = np.append(
            self.sample_values,
            self.function(next_sample),
        )
