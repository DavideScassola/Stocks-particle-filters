import matplotlib.pyplot as plt
import StocksDataUtils as st
from StockModels import *
from MC_inference import *


ticker = 0
start = 0
end = 8000

whole_dataset = st.get_returns(st.stocks, log = True)
selected_data, selected_tickers, seleced_index_list = st.select_no_rounding_stocks(whole_dataset, verbose=True)
print(selected_data.shape)
y_data = selected_data[ticker][start:end]

y_sim, true_x = SVM1_sintetic_data(0.95, 0, 0.2, -0, 3000, burn_in=100)
y_sim, true_x = SVM_smooth_sintetic_data(0.5, -0.40, 0.005, -0, 1000, burn_in=100)

plt.plot(y_sim)
plt.plot(true_x)

print(selected_tickers[48:])
plt.plot(selected_data[6,1000:1200])
st.volatility_clustering_test(y_sim, log=False)
plt.title("volatility clustering on generated data")
#st.plot_tails_comparison(y_data, y_sim)
#plt.show()
#a = np.corrcoef(y_data[0:-1], abs(y_data)[1:])
#print(a)
#st.leverage_effect_test(y_data)
#plt.show()



