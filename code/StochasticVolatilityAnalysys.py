import matplotlib.pyplot as plt
import StocksDataUtils as st
from StockModels import *
from MC_inference import *





###########################################################################################################

ticker = 0
start = 1000
end = 2000

whole_dataset = st.get_returns(st.stocks, log = True)
selected_data, selected_tickers, seleced_index_list = st.select_no_rounding_stocks(whole_dataset, verbose=True)
print(selected_data.shape)
y_data = selected_data[ticker][start:end]


y_sim, true_x = SVM1_sintetic_data(0.98, -0.5, 0.1, -0, 100, burn_in=300)

print("name: ", selected_tickers[ticker])
print("from:{} to:{}".format(st.days[start], st.days[end]))
#plt.plot(data)
"""plt.plot(y_data)
plt.title("y_data")
plt.show()"""

"""
plt.plot(y_sim)
plt.title("y_sim")
st.volatility_clustering_test(y_sim)
plt.show()
"""


###########################################################################################################
y = y_sim

N = 100
particles1, fixed_params1, weights1 = markovian_SMC(SVM1, y, N)
#particles2, fixed_params2, weights2 = anti_decay_markovian_SMC(SVM1, y_sampled, N, minimum_entropy=5, back_jump=1000, v=[0.1, 0.2])
#particles2, fixed_params2, weights2 = LW_filter(SVM1, y, N, h = 0.99, mem=None)
#particles2, fixed_params2, weights2 = LW_filter(SVM1, y_sampled, N, a =0.5, h = 0.9)

"""plt.hist(fixed_params2[:,0][resample(weights2)])
plt.show()
plt.hist(fixed_params2[:,1][resample(weights2)])
plt.show()
plt.hist(fixed_params2[:,2][resample(weights2)])
plt.show()"""

""" 
print(fixed_params2)
plt.hist(fixed_params2[:,0][resample(weights2)], label = "theta")
plt.show()
plt.hist(fixed_params2[:,1][resample(weights2)], label = "mu")
plt.show()
plt.hist(fixed_params2[:,2][resample(weights2)], label = "sigma")
plt.show()
"""
#print(particles.shape)
#plt.plot(particles1.dot(weights1), label = "weighted particles M1", alpha = 0.8)
#plt.plot(particles1[:,0], label = "a particle M1")

#plt.plot(particles2.dot(weights2), label = "weighted particles improv", alpha = 0.8)
plt.plot(particles1, color = "tab:red", alpha = 0.1)
plt.title("100 trajectories")

#plt.plot(true_x, label = "true x", alpha = 0.8)
#plt.plot(SVM1.max_lik_x_pred(y), label = "naive predictor", alpha = 0.5, color = 'pink', linestyle='dashed')
plt.legend()
plt.show()

"""plt.plot(fixed_params1[:,0:3], label = "M1")
plt.plot(fixed_params1[:,0:3], label = "LW")
plt.show()"""


"""
#a = multinomial_resampling(np.array([0.6, 0.1, 0.6]), np.array([1,2,3]))
a = sintetic_data(0.9,-0.8, 0.3, 0.2 ,  10000)
#print(a)

#plt.plot(a)
a = a.reshape([1] + list(a.shape))
print(a.shape)
a = st.return2stocks(a, log=False)
print(a)
plt.plot(a[0])
#plt.hist(a, bins=len(a)//50)

plt.show()

#sampled_params = SMC(prior, mc_sampler, observation_pdf, stocks_series, N)
"""
