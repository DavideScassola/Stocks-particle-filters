import numpy as np
import scipy.stats as stats
import pymc3 as pm
from scipy.linalg import cholesky
import matplotlib.pyplot as plt


def resample(W):
    return np.random.choice(np.arange(len(W)), p=W, size=len(W))

def discrete_entropy(v):
    return len(set(v))

"""
Sequential Monte Carlo algorithm (particle filter) for filering problem
"""
def markovian_SMC(hmm_model, Y, N, mem = None, log_lik_calculation = False):
    T = Y.shape[0]
    x_1, lik_x0 = hmm_model.x_prior_sampler(N, Y[0])
    fixed_params, lik_p0 = hmm_model.prior_fixed_parameters_sampler(N)  # N x n_params
    X = np.empty(shape=[T] + list(x_1.shape))  # T x N
    X[0] = x_1

    W = hmm_model.observation_pdf(Y[0], X[0]) / (lik_p0 * lik_x0)
    W = W / np.sum(W)

    if(mem == None):
        mem = T

    #########################################
    ll = None
    Y_samples = None
    if(log_lik_calculation):
        ll = np.empty(T-1)
        Y_samples = np.empty((T,N))
    #########################################

    for t in range(1, T):

        # resample
        s = max(0, t - mem)
        indexes = resample(W)
        fixed_params = fixed_params[indexes]
        X[s:t] = X[s:t, indexes]

        # propagate
        X[t] = hmm_model.transition_sampler(X[t - 1], fixed_params)

        #########################################
        if(log_lik_calculation):
            ll[t-1] = np.mean((hmm_model.observation_pdf(Y[t], X[t])))

            if(np.isinf(ll[t-1])):
                print("max X:", max(X[t]))
                print("min X:", min(X[t]))
                print("X:", X[t])
                print("Y:", Y[t])
            Y_samples[t] = hmm_model.observation_sampler(X[t])
        #########################################

        # calculate weights
        W = hmm_model.observation_pdf(Y[t], X[t])  # * hmm_model.transition_pdf(X[t], X[t - 1]) / proposal_transition_pdf(X[t], X[t - 1])

        # weight normalization
        W = W / np.sum(W)

    #########################################
    if(log_lik_calculation):
        return X, fixed_params, W, ll, Y_samples
    #########################################

    return X, fixed_params, W




"""
Liu-West particle filter. Particle filter that also estimates the fixed parameters  
"""
def LW_filter(hmm_model, Y, N, h=0.99, mem = None, log_lik_calculation = False):
    T = Y.shape[0]
    x_1, lik_x0 = hmm_model.x_prior_sampler(N, Y[0])
    fixed_params, lik_p0 = hmm_model.prior_fixed_parameters_sampler(N)  # N x n_params
    X = np.empty(shape=[T] + list(x_1.shape))  # T x N
    X[0] = x_1
    W = hmm_model.observation_pdf(Y[0], X[0]) / (lik_p0 * lik_x0)
    W = W / np.sum(W)

    a = np.sqrt(1-(h**2))

    if(mem == None):
        mem = T


    #########################################
    ll = None
    Y_samples = None
    if(log_lik_calculation):
        ll = np.empty(T-1)
        Y_samples = np.empty((T,N))
    #########################################




    for t in range(1, T):
        # print(" ")
        print("\n\nt:", t)
        s = max(0, t - mem)
        # print("starting X[t-1]:", X[t-1])
        # print("starting fixed_params: ", fixed_params)
        # print("W: ", W)

        indexes = resample(W)
        fixed_params = fixed_params[indexes]
        X[s:t] = X[s:t, indexes]


        #########################################
        if(log_lik_calculation):
            Xt = hmm_model.transition_sampler(X[t - 1], fixed_params)
            ll[t-1] = np.mean(hmm_model.observation_pdf(Y[t], Xt))
            Y_samples[t] = hmm_model.observation_sampler(Xt)
        #########################################

        # resample 1

        #m = a * fixed_params + (1 - a) * np.mean(fixed_params, axis = 0)
        # now use m or fixed_params ????
        W = hmm_model.observation_pdf(Y[t], hmm_model.transition_best_guess(X[t - 1], fixed_params))
        W = W / np.sum(W)

        indexes = resample(W)
        fixed_params = fixed_params[indexes]
        X[s:t] = X[s:t, indexes]





        # propagate
        # print("propagation:")

        fixed_params = hmm_model.f_reparametrization(fixed_params)
        m = a * fixed_params + (1 - a) * np.mean(fixed_params, axis=0)
        #print("m:", m[0:2])
        #assert m.shape == (N, 3)
        V = np.cov(fixed_params.T) * h * h
        #print("V:", V)
        #assert V.shape == (3, 3)

        x = np.random.multivariate_normal(mean=np.zeros(3), cov=V, size=N)
        fixed_params = m + x
        fixed_params = hmm_model.b_reparametrization(fixed_params)

        #fixed_params[:, 0][fixed_params[:, 0] > 1] = 1
        fixed_params[:, 2] = abs(fixed_params[:, 2])

        #assert np.all(fixed_params[:, 2] > 0)
        #assert fixed_params.shape == (N, 3)

        """
        #print("fixed_params", fixed_params[0:2])
        plt.hist(fixed_params[:, 0], alpha=0.1, color='b')
        if(t % 100 == 0):
            plt.show()
        """
        X[t] = hmm_model.transition_sampler(X[t - 1], fixed_params)

        # print("sampled X[t]:", X[t])

        # resample 2
        # now use m or fixed_params ????
        #g_x = hmm_model.transition_best_guess(X[t - 1], fixed_params)
        W = hmm_model.observation_pdf(Y[t], X[t])
        W = W / np.sum(W)
        # print("W final resample:", W)


    #########################################
    if(log_lik_calculation):
        return X, fixed_params, W, ll, Y_samples
    #########################################

    return X, fixed_params, W



def anti_decay_markovian_SMC(hmm_model, Y, N, minimum_entropy,
                             back_jump, v):
    T = Y.shape[0]
    x_1, lik_x0 = hmm_model.x_prior_sampler(N, Y[0])
    fixed_params, lik_p0 = hmm_model.prior_fixed_parameters_sampler(N)  # N x n_params
    X = np.empty(shape=[T] + list(x_1.shape))  # T x N
    X[0] = x_1
    W = hmm_model.observation_pdf(Y[0], X[0]) / (lik_p0 * lik_x0)
    W = W / np.sum(W)
    ancestors = np.arange(N)

    def add_noise(p):
        np.random.normal(scale=v[1], size=N)
        n = p
        n[:, 0] = n[:, 0] + np.random.normal(scale=v[0], size=N)
        lik0 = stats.norm.pdf(n[:, 0], p[:, 0], scale=v[0])
        n[:, 1] = n[:, 1] + np.random.normal(scale=v[0], size=N)
        lik1 = stats.norm.pdf(n[:, 1], p[:, 1], scale=v[1])

        return n, lik0 * lik1

    t = 1
    checkpoint = t
    while (t < T):
        # print(" ")
        print(t)
        # print("X[t-1]:", X[t-1])
        # print("fixed_params: ", fixed_params)
        # print("W: ", W)

        # entropy check and eventually restart
        if discrete_entropy(ancestors) < minimum_entropy:
            print("bj")
            t = max(checkpoint, t - back_jump)
            checkpoint = t
            fixed_params, lik = add_noise(fixed_params)
            W = hmm_model.observation_pdf(Y[0], X[0]) / (W * lik)
            W = W / np.sum(W)
            ancestors = np.arange(N)
            print(np.mean(fixed_params, axis = 0))

        # resample
        indexes = resample(W)
        fixed_params = fixed_params[indexes]
        ancestors = ancestors[indexes]
        X[0:t] = X[0:t, indexes]

        # propagate
        X[t] = hmm_model.transition_sampler(X[t - 1], fixed_params)

        # calculate weights
        W = hmm_model.observation_pdf(Y[t], X[t])
        W = W / np.sum(W)

        t = t + 1

    return X, fixed_params, W


"""
Particle filter initialized with MCMC sampled fixed parameters
"""
def MC_filter(hmm_model, Y, N, mem = None, log_lik_calculation = False):
    T = Y.shape[0]
    x_1, lik_x0 = hmm_model.x_prior_sampler(N, Y[0])
    fixed_params = hmm_model.MCMC_prior_parameters
    assert fixed_params.shape == (N, 3)
    lik_p0 = 1
    X = np.empty(shape=[T] + list(x_1.shape))  # T x N
    X[0] = x_1

    W = hmm_model.observation_pdf(Y[0], X[0]) / (lik_p0 * lik_x0)
    W = W / np.sum(W)

    if(mem == None):
        mem = T

    #########################################
    ll = None
    Y_samples = None
    if(log_lik_calculation):
        ll = np.empty(T-1)
        Y_samples = np.empty((T,N))
    #########################################

    for t in range(1, T):

        # resample
        s = max(0, t - mem)
        indexes = resample(W)
        fixed_params = fixed_params[indexes]
        X[s:t] = X[s:t, indexes]

        # propagate
        X[t] = hmm_model.transition_sampler(X[t - 1], fixed_params)

        #########################################
        if(log_lik_calculation):
            ll[t-1] = np.mean((hmm_model.observation_pdf(Y[t], X[t])))

            if(np.isinf(ll[t-1])):
                print("max X:", max(X[t]))
                print("min X:", min(X[t]))
                print("X:", X[t])
                print("Y:", Y[t])
            Y_samples[t] = hmm_model.observation_sampler(X[t])
        #########################################

        # calculate weights
        W = hmm_model.observation_pdf(Y[t], X[t])  # * hmm_model.transition_pdf(X[t], X[t - 1]) / proposal_transition_pdf(X[t], X[t - 1])

        # weight normalization
        W = W / np.sum(W)

    #########################################
    if(log_lik_calculation):
        return X, fixed_params, W, ll, Y_samples
    #########################################

    return X, fixed_params, W


def filter_predictive_evaluation(filter, model, N, y_train_and_test, train_steps):
    particles1, params, weights1, ll, y_samples = filter(model, y_train_and_test, N, mem = 2, log_lik_calculation = True)
    ll = ll[train_steps:]
    y_samples = y_samples[train_steps:]

    assert not np.any(np.isinf(ll))

    mean_log_lik_PF = np.mean(np.log(ll))
    #y_est_PF = np.mean(y_samples, axis = 1)
    return y_samples,   mean_log_lik_PF,   params


def smoothing_evaluation(filter, model, N, y_train_and_test, discard = 0):
    x_samples, params, weights = filter(model, y_train_and_test, N, log_lik_calculation=False, mem = 200)
    return x_samples[discard:], x_samples.dot(weights)[discard:],  params

