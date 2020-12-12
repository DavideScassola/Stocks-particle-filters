import matplotlib.pyplot as plt
import StocksDataUtils as st
from StockModels import *
from MC_inference import *

import pymc3 as pm

tickers = np.array([30,98])
start = 1000
end = start+2000

whole_dataset = st.get_returns(st.stocks, log = True)
selected_data, selected_tickers, selected_index_list = st.select_no_rounding_stocks(whole_dataset, verbose=True)
print(selected_data.shape)
y_data = selected_data[tickers,start:end]

y_sim, true_x = SVM1_sintetic_data(0.95, -0.5, 0.3, -0, 300, burn_in=100)

y = y_data


def make_stochastic_volatility_model(data):
    with pm.Model() as model:
        d = data.shape[0]
        T = data.shape[1]

        m = pm.Normal('m', mu=0, sd=1, shape=T)

        for i in range(d):
            theta = pm.Normal('theta'+str(i), mu=0.5, sd=1)
            sigma = pm.Exponential('sigma'+str(i), 4)
            mu = pm.Normal('mu'+str(i), mu=-2, sd=3)

            sigma_corr = pm.Exponential('sigma_corr'+str(i), 4)
            vec = pm.math.stack((mu, theta))
            volatility = pm.AR('volatility'+str(i), rho=vec, sigma=sigma, constant=True, shape=T)
            y = pm.Normal('returns'+str(i), mu=m*sigma_corr, sd=np.exp(0.5*volatility), observed = data[i])
            #y_sampled = pm.StudentT('returns', nu = nu, sd = np.exp(0.5*volatility), observed = data)
            params = pm.Deterministic('params'+str(i), pm.math.stack((theta, mu, sigma)))

    return model



stochastic_vol_model = make_stochastic_volatility_model(y)

with stochastic_vol_model:
    trace = pm.sample(2000, tune=4000, chains = 2, init='adapt_diag')

with stochastic_vol_model:
    posterior_predictive = pm.sample_posterior_predictive(trace)

#map_estimate = pm.find_MAP(model=stochastic_vol_model)
#print(map_estimate)


pm.traceplot(trace, var_names = ['theta1', 'sigma1', 'mu1', 'sigma_corr1'] )
plt.show()

pm.traceplot(trace, var_names = ['theta0', 'sigma0', 'mu0', 'sigma_corr0'] )
plt.show()


fig, ax = plt.subplots(figsize=(14, 4))

y_vals = np.exp(trace['volatility'])[::5].T

plt.plot(y_vals, 'k', alpha=0.002)
ax.set_ylim(bottom=0)
ax.set(title='Estimated volatility over time', xlabel='Date', ylabel='Volatility')
plt.show()


fig, axes = plt.subplots(nrows=2, figsize=(14, 7), sharex=True)
# y_sim.plot(ax=axes[0], color='black')
axes[1].plot(trace['volatility'][::100].T, 'r', alpha=0.5)
axes[1].plot(true_x, 'b', alpha=1)
axes[1].plot(np.mean(trace['volatility'].T, axis = 1), 'y_sampled', alpha=1)
axes[1].set_title("Posterior volatility")

axes[0].plot(y, color='black')
axes[0].plot(posterior_predictive['returns'][::100].T, 'g', alpha=0.5, zorder=-10)
axes[0].set_title("True log returns (black) and posterior predictive log returns (green)")

plt.show()
