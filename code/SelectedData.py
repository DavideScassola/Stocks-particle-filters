import StocksDataUtils as st

whole_dataset = st.get_returns(st.stocks, log = True)
selected_data, selected_tickers, seleced_index_list = st.select_no_rounding_stocks(whole_dataset, verbose=True)