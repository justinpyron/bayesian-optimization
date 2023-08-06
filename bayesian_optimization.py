import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm



# Parameters
# function: R^1 --> R^1. The function to optimize. It might actually be R^n --> R^1.
# bounds: list of lower/upper bounds within which to optimize.
# theta: TBD



class BayesianOptimizer:
    
    def __init__(
        self,
        function,
        domain_bounds: list[tuple[float, float]],
        theta,
    ):
        self.function = function
        self.domain_bounds = domain_bounds
        self.theta = theta # TODO: rename to something like kernel_bandwidth_penalty or covariance_distance_penalty

        # # Seed it
        self.samples = self.uniformly_sample_domain(number_of_samples=4)
        self.sample_values = np.array([self.function(x) for x in self.samples])
        # self.samples = np.array()
        # self.sample_values = np.array()


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
        return self.sample_values.var()

    # See https://arxiv.org/abs/1807.02811
    def kernel(
        self,
        x: np.array,
        y: np.array,
    ) -> float:
        return self.compute_sample_variance() * np.exp(
            -0.5 * np.square(x - y).sum() / (self.theta ** 2)
        )


    # TODO: make a `normal_distribution` dataclass. https://docs.python.org/3/library/dataclasses.html
    # Give it two attributes: mean (mean_vector), variance (covariance_matrix)


    # Updated August 2023
    def compute_posterior(
        self,
        z: np.array,
    ) -> tuple[np.array, np.array]:
        '''
        Compute the mean and variance of sample point, 
        conditional on samples seen thus far

        If [x_1, x_2] ~ N(mean, var)
        where
            mean = [mean_1, mean_2]
            var = [[var_1_1, var_1_2], [var_2_1, var_2_2]]
        then x_1 | x_2 = observed_values ~ N(mean_tilde, var_tilde)
        where
            mean_tilde = mean_1 + var_1_2 (var_2_2 ^ -1) (a - mean_2)
            var_tilde  = var_1_1 - var_1_2 (var_2_2 ^ -1) var_2_1
        '''

        # TODO: what happens when self.samples is None? --> Ensure that the next sample is uniformly randomly sampled if this is the case

        observed_values = self.sample_values
        mean_1 = observed_values.mean()
        mean_2 = observed_values.mean() * np.ones_like(observed_values)
        var_1_1 = self.kernel(z,z)
        var_1_2 = var_2_1 = np.array([
            self.kernel(z, x) for x in self.samples
        ])
        var_2_2 = np.array([
            [self.kernel(x, y) for x in self.samples] for y in self.samples
        ])

        mean_tilde = mean_1 + var_1_2.dot(
            np.linalg.solve(var_2_2, observed_values - mean_2)
        )
        var_tilde = var_1_1 - var_1_2.dot(
            np.linalg.solve(var_2_2, var_2_1)
        )
        return mean_tilde, var_tilde + 1e-5  # Add small value for numerical stability


    # Updated August 2023
    def compute_expected_improvement(
        self,
        point: np.array,
    ) -> float:
        mean, variance = self.compute_posterior(point)
        st_dev = np.sqrt(variance)
        if st_dev > 1e-5:
            best_yet = self.sample_values.max()
            Z = (mean - best_yet) / st_dev
            return (mean - best_yet)*norm.cdf(Z) + st_dev*norm.pdf(Z)
        else:
            return 0



    def step(self, n):
        '''
        Sample a new point. This point should maximize EI = expected improvement.
        '''
        z = self.uniformly_sample_domain(n)

        posterior = [self.get_posterior(z_i) for z_i in z]
        mu_list = np.array([p[0] for p in posterior])
        var_list = np.array([p[1] for p in posterior])
        EI_list = np.array([self.get_EI(z_i) for z_i in z])
        next_sample = z[np.argmax(EI_list)]
        
        # Sample a new point and add it to set of collected samples
        self.samples = np.append(self.samples, next_sample.reshape((1,self.samples.shape[1])), axis=0)
        self.f = np.append(self.f, self.function(next_sample))
        
        # ------------------------------------------------------------------------
        # TODO: move plotting logic to a different function

        # Plot results (note: visualization only works for 1-D functions)        
        plt.figure(figsize=(20,12))
        ordering_index = np.argsort(z[:,0])
        plt.subplot(2,1,1)
        # Plot true function
        plt.plot(z[ordering_index], 
                 np.array([self.function(z_i) for z_i in z])[ordering_index],
                 c='orange', label='True Function')
        # Plot samples
        plt.scatter(self.samples, self.f, c='orange', s=150)
        # Plot predicted function
        plt.plot(z[ordering_index], 
                 mu_list[ordering_index], 
                 c='dodgerblue', label='Predicted')
        l = mu_list - 1.96 * np.sqrt(var_list)
        u = mu_list + 1.96 * np.sqrt(var_list)
        plt.fill_between(z[ordering_index].squeeze(), 
                         l[ordering_index], 
                         u[ordering_index],
                         color='dodgerblue', alpha=0.2, label='uncertainty bounds')
        # Plot maximum value of samples
        plt.axhline(y=self.f.max(), c='k', linewidth=2, label='max sample')
        # Plot point that maximizes EI
        plt.axvline(x=next_sample, c='k', linewidth=2, linestyle='--', label='next sample location')        
        plt.title('Predicted + True Function')
        plt.legend()
        
        # Plot expected improvement
        plt.subplot(2,1,2)
        plt.plot(z[ordering_index], EI_list[ordering_index], c='green', label='EI')
        plt.title('Expected Improvement')
        plt.show()
        # ------------------------------------------------------------------------
        
    