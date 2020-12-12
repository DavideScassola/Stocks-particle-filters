import StocksDataUtils as st
from PlotUtils import quantiles_plot
from StockModels import *
from MC_inference import *
import SelectedData as sd



#############################################################################
# data selection
ticker = 48
series_length = 3000

end = sd.selected_data.shape[1]
start = end - series_length
ticker_name = sd.selected_tickers[ticker]
start_date = st.days[start]
end_date = st.days[end]

y_data = sd.selected_data[ticker, start:end]

# trainig set - test set separation
train_steps = 500
y_train_data = sd.selected_data[ticker, start-train_steps:start]
y_train_and_test = sd.selected_data[ticker, start-train_steps:end]

# plot data
title = "{} log-returns from: {} to {}".format(ticker_name, start_date, end_date)
print(title)
plt.plot(y_data, label = title)
plt.legend()
plt.title(title)

# plot data features
plt.hist(y_data, bins=100, label = "IBM log-returns")

plt.plot(abs(y_data), label = "IBM absolute value of log-returns")
plt.legend()
st.volatility_clustering_test(y_data, log=False)
st.autocorrelation_test(y_data, verbose=True)
#############################################################################




#############################################################################
# for each model

MCMC_train_parameters_trace = SVM1.MCMC_parameters(y_data, samples = 1000, tune = 1000, chains = 2)
# testing predictivity on true data
N = 10000 # number of particles

SVM1.MCMC_prior_parameters = MCMC_train_parameters_trace[np.random.choice(np.arange(MCMC_train_parameters_trace.shape[0]), N)]

y_samples_naive, mean_log_lik_naive, params_NG = naive_gaussian_fit(y_train_and_test, N, train_steps)
y_samples_PF,   mean_log_lik_PF,   params_PF   = filter_predictive_evaluation(markovian_SMC, SVM1, N, y_train_and_test, train_steps)
y_samples_LWPF, mean_log_lik_LWPF, params_LWPF = filter_predictive_evaluation(LW_filter, SVM1, N, y_train_and_test, train_steps)
y_samples_MCPF, mean_log_lik_MCPF, params_MCPF = filter_predictive_evaluation(MC_filter, SVM1, N, y_train_and_test, train_steps)

# plot predictivity
print("mean_log_lik_naive",  mean_log_lik_naive)
print("mean_log_lik_PF"  ,   mean_log_lik_PF   )
print("mean_log_lik_LWPF" ,  mean_log_lik_LWPF )
print("mean_log_lik_MCPF"  , mean_log_lik_MCPF )

quantiles_plot(y_data, y_samples_naive, areas = [0.99, 0.9, 0.5], label = "y_est_naive", lw = 0.6)
quantiles_plot(y_data, y_samples_PF, areas = [0.99, 0.9, 0.5], label = "PF", lw = 0.6)
quantiles_plot(y_data, y_samples_LWPF, areas = [0.99, 0.9, 0.5], label = "LWPF", lw = 0.6)
quantiles_plot(y_data, y_samples_MCPF, areas = [0.99, 0.9, 0.5], label = "MCPF", lw = 0.6)

# plot learned parameters

plt.hist(params_LWPF[:,0], label="phi", bins = 100); plt.legend()
plt.hist(params_LWPF[:,1], label="mu", bins = 100); plt.legend()
plt.hist(params_LWPF[:,2], label="sigma", bins = 100); plt.legend()

for i in params_PF.shape[1]:
    plt.hist(params_PF, label="params_PF")
    plt.hist(params_LWPF, label="params_LWPF")
    plt.hist(params_MCPF, label="params_MCPF")


# generate synthetic data
parameters = [0.95, -0.5, 0.3, 0]
y_sim, true_x = SVM1_sintetic_data(0.95, -0.5, 0.3, 0, 2000, burn_in=100)

# plot synthetic data
plt.plot(y_sim, label = "synthetic data")
plt.legend()

# testing smoothing on synthetic data
warm_up = 1000
x_samples_PF, x_est_PF,  params_PF = smoothing_evaluation(markovian_SMC, SVM1, N, y_sim, warm_up)
x_samples_LWPF, x_est_LWPF, params_LWPF = smoothing_evaluation(LW_filter, SVM1, N, y_sim, warm_up)

# Smoothing with MCMC
MCMC_trace = SVM1.MCMC(y_sim, samples = 2000, tune = 4000, chains = 2)
x_samples_MCMC = MCMC_trace['volatility'][:,warm_up:] # samples x Time
x_est_MCMC = np.mean(x_samples_MCMC, axis = 0)
params_MCMC = MCMC_trace['params'] # samples x param dim (3)


# plot smoothing
print("PF smoothing error",   np.std(x_est_PF-true_x[warm_up:]))
print("LWPF smoothing error", np.std(x_est_LWPF-true_x[warm_up:]))
print("MCMC smoothing error", np.std(x_est_MCMC-true_x[warm_up:]))
print("single MCMC smoothing error", np.std(x_samples_MCMC[9]-true_x[warm_up:]))


plt.plot(x_samples_MCMC[0], label = "single MCMC", alpha = 0.6)

plt.plot(x_est_MCMC, label = "average MCMC", alpha = 0.8)

plt.plot(x_est_LWPF, label = "LW", alpha = 0.6)

plt.plot(x_est_PF, label = "PF", alpha = 0.6)
plt.plot(true_x[warm_up:], label = "true volatility", alpha = 0.6)
plt.legend()

# plot parameters
plt.hist(params_LWPF[:,0], label="phi LW", bins = 100, alpha = 0.6, density = True)
plt.hist(params_MCMC[:,0], label="phi MCMC", bins = 100, alpha = 0.6, density = True)
plt.legend()

plt.hist(params_LWPF[:,1], label="mu LW", bins = 100, alpha = 0.6, density = True)
plt.hist(params_MCMC[:,1], label="mu MCMC", bins = 100, alpha = 0.6, density = True)
plt.legend()

plt.hist(params_LWPF[:,2], label="sigma LW", bins = 100, alpha = 0.6, density = True)
plt.hist(params_MCMC[:,2], label="sigma MCMC", bins = 100, alpha = 0.6, density = True)
plt.legend()


fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
p = 0
axes[0].hist(params_PF[:,p], label="params_PF", bins = 50)
axes[0].legend()
axes[1].hist(params_LWPF[:,p], label="params_LWPF", bins = 50)
axes[1].legend()
axes[2].hist(params_MCMC[:,p], label="params_MCMC", bins = 50)
axes[2].legend()


plt.show()

