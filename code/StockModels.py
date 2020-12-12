from HMM_Utils import *
import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt


def SVM1_observation_pdf(y, x_current):
    # Xt -> Yt
    return stats.norm.pdf(y, scale=np.exp(x_current / 2))


def SVM1_X_transition_pdf(x_current, x_prev, params):
    # X_t -> X_t+1

    # theta:0, mu:1, std:2
    loc = params[:, 1] + params[:, 0] * x_prev
    scale = params[:, 2]
    return stats.norm.pdf(x_current, loc, scale)

def SVM1_X_transition_best_guess(x_current, params):
    # theta:0, mu:1, std:2
    return params[:, 1] + params[:, 0] * x_current


def SVM1_X_transition_sampler(x_prev, params):
    # X_t -> X_t+1
    # theta:0, mu:1, std:2
    loc = params[:, 1] + params[:, 0] * x_prev
    scale = params[:, 2]
    return np.random.normal(loc, scale)

def SVM1_observation_sampler(x):
    return np.random.normal(scale=np.exp(x/2))


def SVM1_f_reparametrization(params):
    r = params
    r[:,2] = np.log(r[:,2])
    return r

def SVM1_b_reparametrization(params):
    r = params
    r[:,2] = np.exp(r[:,2])
    return r


def SVM1_x_prior_sampler(N, y, std = 3):
    #mu = 2*np.log(abs(y_sampled))
    mu = SVM1_naive_predictor(y)
    sample = np.random.normal(mu, std, size=N)
    lik = stats.norm.pdf(sample, loc=mu, scale=std)
    return sample, lik

def SVM1_prior_fixed_parameters_sampler(N):
    # theta:0, mu:1, std:2
    p = np.empty(shape = (N,3))

    """
    p[:,0] = np.random.uniform(0.88,0.92,size=N) #np.abs(np.random.normal(0.8,0.3,size=N))
    p[:,1] = np.random.uniform(-1,-0.7,size=N)
    p[:,2] = np.random.uniform(0.29,0.31,size=N) #np.exp(np.random.normal(0,1,size=N))
    """

    p[:,0] = np.random.uniform(-1,1,size=N) #np.abs(np.random.normal(0.8,0.3,size=N))
    p[:,1] = np.random.uniform(-4,4,size=N)
    p[:,2] = np.random.uniform(0.01,4,size=N) #np.exp(np.random.normal(0,1,size=N))

    return p, 1

def make_SVM1_mcmc_model(data):
    with pm.Model() as model:
        theta = pm.Normal('theta', mu=0.5, sd=1)
        sigma = pm.Exponential('sigma', 4)
        mu = pm.Normal('mu', mu=-2, sd=3)
        #nu = pm.Exponential('nu', 0.1)
        vec = pm.math.stack((mu, theta))

        volatility = pm.AR('volatility', rho=vec, sigma=sigma, constant=True, shape=len(data))
        y = pm.Normal('returns', mu=0, sd=np.exp(0.5*volatility), observed = data)
        #y_sampled = pm.StudentT('returns', nu = nu, sd = np.exp(0.5*volatility), observed = data)
        params = pm.Deterministic('params', pm.math.stack((theta, mu, sigma)))

    return model


def SVM1_MCMC_parameters(y, samples = 2000, tune = 4000, chains = 2):
    stochastic_vol_model = make_SVM1_mcmc_model(y)
    with stochastic_vol_model:
        s = pm.sample(samples, tune=tune, chains = chains, init='adapt_diag')
        pm.traceplot(s, var_names=['theta', 'sigma', 'mu'])
        plt.show()
    return s['params']


def SVM1_MCMC(y, samples = 2000, tune = 4000, chains = 2):
    stochastic_vol_model = make_SVM1_mcmc_model(y)
    with stochastic_vol_model:
        s = pm.sample(samples, tune=tune, chains = chains, init='adapt_diag')
        pm.traceplot(s, var_names=['theta', 'sigma', 'mu'])
        plt.show()
    return s


SVM1 = HMM(SVM1_observation_pdf, SVM1_X_transition_pdf, SVM1_X_transition_sampler, SVM1_X_transition_best_guess,
           SVM1_f_reparametrization, SVM1_b_reparametrization, SVM1_observation_sampler, SVM1_x_prior_sampler,
           SVM1_prior_fixed_parameters_sampler)

SVM1.MCMC_parameters = SVM1_MCMC_parameters
SVM1.MCMC = SVM1_MCMC


"""
x_t = theta*x_t-1 + N(mu_1, std_1)
y_t = exp(x_t/2)*N(0,1)
"""
def SVM1_sintetic_data(theta, mu, std, x_0, T, burn_in = 0):
    #assert burn_in < T
    T = burn_in + T
    x = np.zeros(T)
    x[0] = x_0
    eps = np.random.normal(loc=mu,scale=std,size=T)
    for i in range(1,len(x)):
        x[i] = theta*x[i-1] + eps[i]

    y = np.exp(x/2)*np.random.normal(size=T)

    return y[burn_in:], x[burn_in:]


"""
x_t = theta*x_t-1 + N(mu_1, std_1)
y_t = exp(x_t/2)*N(0,1)
"""
def SVM_smooth_sintetic_data(theta, mu, std, x_0, T, burn_in = 0):
    #assert burn_in < T
    T = burn_in + T
    x = np.zeros(T)
    step = np.zeros(T)
    x[0] = x_0
    eps = np.random.normal(loc=0,scale=std,size=T)

    for i in range(1,len(x)):
        step[i] = theta*step[i-1] + eps[i]

    for i in range(1,len(x)):
        x[i] = x[i-1] + step[i] + mu

    y = np.exp(x/2)*np.random.normal(size=T)

    return y[burn_in:], x[burn_in:]



def SVM1_naive_predictor(y):
    return 2*np.log(abs(y))     #W R O N G !
    #return np.log((y_sampled*y_sampled))




def naive_gaussian_fit(y, N, t):
    mean = np.mean(y)
    std = np.std(y)

    return np.random.normal(mean, std, size = N*(len(y)-t)).reshape(len(y)-t, -1), np.mean(stats.norm.logpdf(y, loc=mean, scale=std)), [mean, std]









"""
parameters = (0.9, -1, 1)
x = np.arange(4)

a = SVM1.transition_sampler(x, parameters)
b = SVM1.observation_pdf(1, x)
c = SVM1.transition_pdf(1, x, parameters)
print(a, b, c)
"""


