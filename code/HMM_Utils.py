import numpy as np
import scipy


class HMM:
    def __init__(self, observation_pdf, transition_pdf, transition_sampler, transition_best_guess, f_reparametrization, b_reparametrization, observation_sampler, x_prior_sampler, prior_fixed_parameters_sampler):
        self.observation_pdf = observation_pdf
        self.transition_pdf = transition_pdf
        self.transition_sampler = transition_sampler
        self.transition_best_guess = transition_best_guess
        self.f_reparametrization = f_reparametrization
        self.b_reparametrization = b_reparametrization
        self.observation_sampler = observation_sampler
        self.x_prior_sampler = x_prior_sampler
        self.prior_fixed_parameters_sampler = prior_fixed_parameters_sampler

    def approximate_likelihood(self, y, x_prev_samples):
        x_current_samples = self.transition_sampler(x_prev_samples)
        return np.mean(self.observation_pdf(y, x_current_samples))

    def max_lik_x_pred(self, y):
        result = np.empty(len(y))
        for i in range(len(y)):
            f = lambda x: -self.observation_pdf(y[i], x)
            res = scipy.optimize.minimize(f, x0 = 1, tol=1e-3)
            result[i] = res.x
        return result





