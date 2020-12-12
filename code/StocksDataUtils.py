import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import pickle

with open("../stocks_1990_2019_few_holes.pk", 'rb') as path:
    data = pickle.load(path)
stocks, tickers, ticker_to_pos, days, day_to_pos, info = data

interesting_value = 1  # 1:'adj_close' 0:'close'
stocks = stocks[:, :, interesting_value]

"""
Extracts the return series given the stock values series
with n_days you can choose to aggregate the returns of n consecutive days (ex: 5 gives weekly returns)
"""


def get_returns(stocks, log=True, n_days=None):
    tot_tickers, tot_days = stocks.shape
    returns = np.zeros((tot_tickers, tot_days - 1))
    for d in range(1, tot_days):
        returns[:, d - 1] = stocks[:, d] / stocks[:, d - 1]

    if n_days is not None:
        n_groups = returns.shape[1] // n_days
        aggregated_returns = np.zeros((returns.shape[0], n_groups))
        for i in range(n_groups):
            aggregated_returns[:, i] = np.prod(returns[:, i * n_days:(i + 1) * n_days], axis=1)
        returns = aggregated_returns

    if (log):
        returns = np.log(returns)

    return returns


"""
Normalizes the given log-return dataset
"""


def normalize(returns):
    mean = np.mean(returns)
    std = np.std(returns)
    normalized_returns = (returns - mean) / std

    def denormalize(normal_returns):
        return normal_returns * std + mean

    return normalized_returns, denormalize


"""
Filters from the returns dataset the tickers that have at maximum 1/max_ratio number of zero values, i.e.
that ones that have a few amount of days with low volume of exchanges
"""


def select_no_rounding_stocks(whole_dataset, max_ratio_of_zeros=28, verbose=False):
    count = 0
    index_list = []
    selected_ticker_list = []

    for i in range(len(whole_dataset)):
        r = whole_dataset[i]
        zeroday = len(r[r == 0])
        # print(zeroday)
        if (zeroday < whole_dataset.shape[1] // max_ratio_of_zeros):  # 28 -> 100 tickers
            index_list += [i]
            selected_ticker_list += [tickers[i]]
            count += 1

    if (verbose):
        print(count, "tickers selected")
        print(selected_ticker_list)

    return whole_dataset[index_list], selected_ticker_list, index_list


"""
Splits the dataset into train, validation and test
(the validation Y is taken from the train Y)
"""


def dataset_splitter(dataset, train_prop=0.90, val_prop=0.1):
    tot_tickers, tot_len = dataset.shape

    split_point2 = int(tot_len * train_prop)
    split_point1 = int(split_point2 * (1 - val_prop))

    train_data = dataset[:, 0:split_point1]
    val_data = dataset[:, split_point1:split_point2]
    test_data = dataset[:, split_point2:]

    return train_data, val_data, test_data


"""
Checks if there is linear autocorrelation in the given returns series

max_lag: maximum number of lag for wich the autocorrelation is calculated
threshold: value above which the test is considered failed
verbose: if True it plots the result
label: label for the plot
"""


def autocorrelation_test(single_stock_series, threshold=0.1, max_lag=100, verbose=True, label=None):
    cor = []
    passed = True

    for k in range(1, max_lag):
        c = np.corrcoef(single_stock_series[0:-k], single_stock_series[k:])[0, 1]
        cor += [c]
        passed = passed and np.abs(c) < threshold

    if (verbose):
        print("correlations: ", cor)
        plt.title("linear unpredictability")
        plt.xlabel("lag k")
        plt.ylabel("auto-correlation")
        _ = plt.plot(cor, label=label)
        plt.legend()

    return passed


np.seterr(divide='ignore', invalid='ignore')

"""
Plots a log-log histogram for checking if there is a power law decay in the tails.
The histogram plots the distribution of the absolute values of returns (volatility).
Returns true if the coefficient of the power law decay (alpha) is in the interval [3,5] (check the paper)

calculate_alpha: if True returns an estimate of the coefficient of the power law decay
verbose: if True it plots the result
label: label for the plot
"""


def fat_tails_test(single_stock_series, calculate_alpha=True, normalize=False, label=None, verbose=False):
    data = single_stock_series

    if (normalize):
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std

    data = np.abs([data])
    data = data[data != 0]  # avoid numerical problems

    if (verbose):
        plt.title("heavy tailed distribution")
        plt.ylabel("density")
        plt.xlabel("log-return")
        b = range(-2, 3)
        plt.xticks(b, ["10e%s" % i for i in b])
        density = plt.hist(np.log10(data), log=True, density=True, bins=50, range=(-2, 1), label=label)
        plt.legend()
    else:
        density = np.histogram(np.log10(data), range=(-2, 1), bins=50, density=True)

    if (calculate_alpha):
        results = powerlaw.Fit(data)
        return results.power_law.alpha > 3 and results.power_law.alpha < 5


"""  # calculating alpha: power low decay coefficient
  if(calculate_alpha):
    Y = density[0]
    x = density[1]

    tail_begin = 0
    tail_end = 0

    for counter, value in enumerate(x):
      if(value>=-1.2): # the starting point of the tail
        tail_begin = counter
        break

    for counter, value in enumerate(Y):
      if(value==0):
        tail_end = counter-4 # because Y near to the end is noisy
        break


    Y = Y[tail_begin:tail_end]
    x = x[tail_begin:tail_end]
    Y = np.log10(Y)


    alpha = -np.polyfit(x, Y, 1)[0]
    print("\nalpha:", alpha)"""

"""
Checks if there is linear autocorrelation in the volatility from the given returns series.
Returns true if for the first max_lag days the estimated beta (check paper) is in the interval [0.1,0.5] 

max_lag: maximum number of lag for wich the autocorrelation is calculated
verbose: if True it plots the result
log: if true plots a loglog plot
label: label for the plot
returncorr: if true returns the correlations
"""


def volatility_clustering_test(single_stock_series, max_lag=100, log=True, verbose=True, label="", returncorr=False):
    cor = []
    volatility = np.abs(single_stock_series)

    for k in range(1, max_lag):
        c = np.corrcoef(volatility[0:-k], volatility[k:])[0, 1]
        cor += [c]

    beta = -np.polyfit(np.log(np.arange(1, max_lag)), np.log(cor), 1)[0]

    if (verbose):
        print("correlations: ", cor)

        plt.title("volatility clustering")
        plt.xlabel("lag k")
        plt.ylabel("auto-correlation")
        if (log):
            _ = plt.loglog(cor, label=label)
        else:
            _ = plt.plot(cor, label=label)
        plt.legend()
        print("beta:", beta)
    if (returncorr):
        return cor
    else:
        return (beta > 0.1 and beta < 0.5)


"""
Checks for the presence of leverage effect from the given returns series.
Returns true if for the first 10 days the correlation is negative

max_lag: maximum number of lag for wich the correlation is calculated
verbose: if True it plots the result
label: label for the plot
correlations: if true shows correlation, if false the L statistic (check paper)
"""


def leverage_effect_test(single_stock_series, max_lag=100, verbose=True, correlations=True, label=""):
    L = []
    r = single_stock_series
    r2 = r ** 2
    r3 = r ** 3
    volatility = np.abs(single_stock_series)

    if (not correlations):
        for k in range(1, max_lag + 1):
            l = np.mean(r[0:-k] * r2[k:] - r3[0:-k]) / (np.mean(r2)) ** 2
            L += [l]
    else:
        for k in range(1, max_lag + 1):
            l = np.corrcoef(r[0:-k], volatility[k:])[0, 1]
            L += [l]

    if (verbose):
        if (not correlations):
            plt.ylabel("L")
        else:
            plt.ylabel("correlation")
        print("L(k): ", L)
        plt.title("leverage effect")
        plt.xlabel("lag k")
        _ = plt.plot(L, label=label)
        plt.legend()

    return (np.array(L) < 0).all()


"""
calculates the mean likelihood of Y in the train and validation datasets using as model
a multivariate Gaussian whose parameters are estimated by the training set 
Y shape: (n_tickers, n_days)
"""


def mean_likelihood(train_data, val_data):
    from scipy.stats import multivariate_normal

    # td = train_data[:,~np.all(train_data == 0, axis=1)]
    # vd = val_data[:,~np.all(val_data == 0, axis=1)]

    mean = np.mean(train_data, axis=1)
    cov = np.cov(train_data)

    train_likelihood = np.mean(multivariate_normal.logpdf(train_data.T, mean, cov) / mean.shape[0])
    val_likelihood = np.mean(multivariate_normal.logpdf(val_data.T, mean, cov) / mean.shape[0])
    print("train likelihood: ", train_likelihood)
    print("validation likelihood: ", val_likelihood)


"""
It returns the returns matrix but removing the days in which the mean volatility is lower than a given threshold
"""


def volatility_filter(data, threshold=0.02):
    v = np.abs(data)
    vm = np.mean(v, axis=0)
    vf = data[:, vm >= threshold]
    return vf


"""
It compares the correlations of two given datasets
it shows the mean l1 distance between the two matrices and also the correlation between them 
"""


def vertical_correlation_test(true_dataset, simulated_dataset, volatility_threshold=None):
    true = true_dataset
    simulated = simulated_dataset

    if (volatility_threshold != None):
        true = volatility_filter(true_dataset, threshold=volatility_threshold)
        simulated = volatility_filter(simulated_dataset, threshold=volatility_threshold)

    cor_true = np.corrcoef(true)
    cor_sim = np.corrcoef(simulated)

    mean_error = np.mean(np.abs((cor_true - cor_sim)))

    print("\ncorrelations comparison")
    print("mean error: {0}".format(mean_error))
    print("correlations correlation: ", np.corrcoef(cor_true.reshape(-1), cor_sim.reshape(-1))[0, 1])


"""
It compares the correlations in volatilities of two given returns series
it shows the mean l1 distance between the two correlation series and also the correlation between them 
"""


def volatility_clustering_comparison(true_series, simulated_series, max_lag=100):
    cor_true = volatility_clustering_test(true_series, max_lag=max_lag, log=False, verbose=False, returncorr=True)
    cor_sim = volatility_clustering_test(simulated_series, max_lag=max_lag, log=False, verbose=False, returncorr=True)
    cor_true = np.array(cor_true)
    cor_sim = np.array(cor_sim)

    mean_error = np.mean(np.abs((cor_true - cor_sim)))

    print("\nvolatility clustering comparison")
    print("mean error: {0}".format(mean_error))
    print("volatility correlations correlation: ", np.corrcoef(cor_true, cor_sim)[0, 1])


"""
It compares the histograms of returns and of the tails between two returns series
it shows the mean l1 distance between the two histograms and also the correlation between them 
"""


def distribution_comparison(true_data, simulated_dataset, bins=50):
    true_dataset = true_data

    true_density = np.histogram(true_dataset, bins=bins, density=True, range=[-0.1, 0.1])[0]
    simulated_density = np.histogram(simulated_dataset, bins=bins, density=True, range=[-0.1, 0.1])[0]

    vol_true_density = np.histogram(np.abs(true_dataset), bins=bins, density=True)[0]
    vol_simulated_density = np.histogram(np.abs(simulated_dataset), bins=bins, density=True)[0]

    mean_error = np.mean(np.abs((true_density - simulated_density)))
    vol_mean_error = np.mean(np.abs((vol_true_density - vol_simulated_density)))

    print("\ndistribution comparison")
    print("mean error: {0}".format(mean_error))
    print("density correlation: ", np.corrcoef(true_density, simulated_density)[0, 1])

    # print(true_density)
    # print(simulated_density)
    # print(true_density-simulated_density)

    print("\nvolatility distribution comparison")
    print("mean error: {0}".format(vol_mean_error))
    print("density correlation: ", np.corrcoef(vol_true_density, vol_simulated_density)[0, 1])


"""
plots the histograms of the two given return series
"""


def plot_density_comparison(true_data, simulated_data, bins=50):
    plt.title("distribution")
    density = plt.hist(true_data, log=False, bins=bins, range=[-0.1, 0.1], alpha=0.5, density=True, label="true Y")
    density = plt.hist(simulated_data, log=False, bins=bins, range=[-0.1, 0.1], alpha=0.5, density=True,
                       label="generated Y")
    plt.legend()


"""
plots the histograms of the tails of the two given return series
"""


def plot_tails_comparison(true_data, simulated_data, bins=50, axis=None):
    where = plt
    if (axis != None):
        where = axis

    where.xlabel("log-return")
    b = range(-2, 3)
    plt.xticks(b, ["10e%s" % i for i in b])
    data = np.abs([true_data])
    data = data[data != 0]  # avoid numerical problems
    density = where.hist(np.log10(data), log=True, bins=bins, range=(-2, 1), alpha=0.5, density=True, label="true Y")

    data = np.abs([simulated_data])
    data = data[data != 0]  # avoid numerical problems
    density = where.hist(np.log10(data), log=True, bins=bins, range=(-2, 1), alpha=0.5, density=True,
                         label="generated Y")

    plt.legend()


"""
it produces a stock series given a log-returns series, 
starting from a initial stock price of 1 for all the tickers (should be change in order to choose the initial stock price)
log: if true it returns the logarthm of the stocks
"""


def return2stocks(returns_series, log=True):
    stocks = np.zeros((returns_series.shape[0], returns_series.shape[1] + 1))
    for i in range(0, returns_series.shape[1]):
        stocks[:, i + 1] = returns_series[:, i] + stocks[:, i]

    if (log):
        return stocks
    else:
        return np.exp(stocks)


"""
From a given return series it computes the histograms of the time you have to wait in order to see a gain/loss of theta (check paper)

verbose: if True it plots the result
log_x: if true the x axis of the plot is in log10 scale
"""


def gain_loss_asymmetry_test(single_stock_series, theta=0.1, log_x=False, bins=20, verbose=True, axis=None):
    def T_t_wait(stock, t, theta):
        max_waiting_time = len(stock)
        waiting_negative = 1
        done_neg = False
        waiting_positive = 1
        done_pos = False

        for waiting_time in range(t, max_waiting_time):
            if ((not (stock[waiting_time] - stock[t]) > theta) and not done_pos):
                waiting_positive += 1
            else:
                done_pos = True

            if ((not (stock[waiting_time] - stock[t]) < -theta) and not done_neg):
                waiting_negative += 1
            else:
                done_neg = True

            if (done_neg and done_pos):
                return ([waiting_positive], [waiting_negative])

        g = []
        l = []
        if (done_pos):
            g = [waiting_positive]
        if (done_neg):
            l = [waiting_negative]

        return (g, l)

    max_waiting_time = len(single_stock_series)
    gain_times = []
    loss_times = []
    stocks = return2stocks(single_stock_series.reshape((1, max_waiting_time)), log=True)[0, :]
    for t in range(1, max_waiting_time):
        g, l = T_t_wait(stocks, t, theta)
        gain_times += g
        loss_times += l

    if (log_x):
        gain_times = np.log10(gain_times)
        loss_times = np.log10(loss_times)

        if (verbose):
            where = plt
            if (axis != None):
                where = axis

            xlabel = "return time"
            ylabel = "density"
            where.title("gain/loss asymmetry")
            where.xlabel(xlabel)
            where.ylabel(ylabel)
            density = where.hist((gain_times), log=False, alpha=0.5, density=True, label="gain", bins=bins)
            density = where.hist((loss_times), log=False, alpha=0.5, density=True, label="loss", bins=bins)
            plt.legend()

    return gain_times, loss_times


def coarse_volatility(single_return_series, tau=5):
    n = len(single_return_series)
    r = single_return_series.reshape((1, n))
    stocks = return2stocks(r, log=True)[0, :]
    cv_vector = np.zeros((n - tau))

    for i in range(len(cv_vector)):
        cv_vector[i] = stocks[i + tau] - stocks[i]

    cv_vector = np.abs(cv_vector)

    return cv_vector


def fine_volatility(single_return_series, tau=5):
    n = len(single_return_series)
    fv_vector = np.zeros((n - tau))
    volatility = np.abs(single_return_series)

    for i in range(len(fv_vector)):
        fv_vector[i] = np.sum(volatility[i:i + tau])

    return fv_vector


def coarse_fine_volatility_correlation(single_return_series, max_lag=20, tau=5):
    cv = coarse_volatility(single_return_series, tau=tau)
    fv = fine_volatility(single_return_series, tau=tau)

    p_cf = []

    # print("ciao", max_lag, "   ", np.corrcoef((cv[0:-max_lag], fv[max_lag:]))[0,1] )

    for k in range(1, max_lag + 1):
        c = np.corrcoef(cv[0:-k], fv[k:])[0, 1]
        p_cf += [c]

    p_cf = p_cf[::-1]
    c = np.corrcoef(fv, cv)[0, 1]
    p_cf += [c]

    for k in range(1, max_lag + 1):
        c = np.corrcoef(fv[0:-k], cv[k:])[0, 1]
        p_cf += [c]

    return p_cf


"""
From a given return series shows coarse_fine volatility correlation plot (check paper)
It returns true if for the first 4 days the delta is negative (an empirical and easy way to check if roughly this statistical property is present)

verbose: if True it plots the result
"""


def coarse_fine_volatility_correlation_test(single_return_series, max_lag=20, tau=5, verbose=True, axis=None):
    p_cf = coarse_fine_volatility_correlation(single_return_series, max_lag=max_lag, tau=tau)

    delta_p_cf = []
    start = max_lag
    assert p_cf
    for k in range(0, max_lag + 1):
        delta_p_cf += [p_cf[start + k] - p_cf[start - k]]

    if (verbose):
        range1 = np.arange(-max_lag, max_lag + 1)
        range2 = np.arange(0, max_lag + 1)

        where = plt  # ok this is bad
        if (axis != None):
            where = axis

        xlabel = "lag k"
        ylabel = "correlation"
        where.title("coarse-fine volatility correlation")
        where.plot([-max_lag, max_lag], [0, 0])
        where.plot(range1, p_cf, '.', label="cf_vol_corr")
        where.plot(range2, delta_p_cf, '.', label="delta")
        where.xlabel(xlabel)
        where.ylabel(ylabel)
        where.legend()

    return ((np.array(delta_p_cf[1:4])) < 0).all()


"""
Given a dataset of return series it checks for the presence of several statistical properties, 
showing for each test how many tickers respect that property.

slow_stats: if true it checks also for properties that are very computationally intensive to check, (for now just fat tails, gain-loss asymmetry is even slower and not present here)
for example checking for the presence of fat tails distribution: unlikely compute alpha requires lot of time.
For the other statistics for examples it takes less than 20s for a dataset of 100 tickers x 6000 days
"""


def total_test(return_dataset, slow_stats=False):
    def linear_unpredictability(r):
        return autocorrelation_test(r, threshold=0.1, max_lag=20, verbose=False, label=None)

    def fat_tails(r):
        return fat_tails_test(r, verbose=False)

    def leverage_effect(r):
        return leverage_effect_test(r, max_lag=10, verbose=False, correlations=True)

    def volatility_clustering(r):
        return volatility_clustering_test(r, max_lag=150, verbose=False)

    def coarse_fine_volatility(r):
        return coarse_fine_volatility_correlation_test(r, max_lag=6, tau=5, verbose=False)

    test_functions = [linear_unpredictability, leverage_effect, volatility_clustering, coarse_fine_volatility]
    if (slow_stats):
        test_functions += [fat_tails]

    n_tickers = return_dataset.shape[0]
    n_tests = len(test_functions)
    results = np.zeros((n_tickers, n_tests))

    for t in range(n_tickers):
        for i, test in enumerate(test_functions):
            results[t, i] = test(return_dataset[t])

    # print(results)
    accumulated_result = np.sum(results, axis=0)

    for i, test in enumerate(test_functions):
        print("{}: {}/{}".format(test.__name__, accumulated_result[i], n_tickers))

    return accumulated_result, results
