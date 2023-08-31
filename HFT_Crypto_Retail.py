# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:08:25 2023

@author: Fayssal
"""
import websocket
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create empty DataFrame to store incoming data
df_ethusdt = pd.DataFrame(columns=['symbol', 'bids', 'asks', 'timestamp'])
df_btcusdt = pd.DataFrame(columns=['symbol', 'bids', 'asks', 'timestamp'])
df_ethbtc = pd.DataFrame(columns=['symbol', 'bids', 'asks', 'timestamp'])

counter = 0  # Add counter to control heatmap rendering frequency

def plot_order_book_heatmap(bids, asks):
    bid_df = pd.DataFrame(bids, columns=["Price", "Quantity"])
    ask_df = pd.DataFrame(asks, columns=["Price", "Quantity"])

    bid_df["Price"] = bid_df["Price"].astype(float)
    bid_df["Quantity"] = bid_df["Quantity"].astype(float)
    ask_df["Price"] = ask_df["Price"].astype(float)
    ask_df["Quantity"] = ask_df["Quantity"].astype(float)

    bid_df["Type"] = "Bid"
    ask_df["Type"] = "Ask"
    order_book_df = pd.concat([bid_df, ask_df])

    plt.figure(figsize=(10, 8))
    sns.heatmap(order_book_df.pivot_table(index="Price", columns="Type", values="Quantity", aggfunc="sum").fillna(0),
                annot=True, cmap="coolwarm", cbar=True)
    plt.title("Order Book Heatmap")
    plt.show()

def on_message(ws, message):
    global df_ethusdt, df_btcusdt, df_ethbtc, counter  # Update to include counter

    data = json.loads(message)

    if "e" in data and data["e"] == "depthUpdate":
        symbol = data['s']
        bids = data['b']
        asks = data['a']
        timestamp = data['E']

        if symbol == 'ETHUSDT':
            df_ethusdt = df_ethusdt.append({'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': timestamp}, ignore_index=True)
            counter += 1  # Increment counter
            if counter % 50 == 0:  # Display heatmap every 50 messages
                bids = df_ethusdt.iloc[-1]['bids']
                asks = df_ethusdt.iloc[-1]['asks']
                plot_order_book_heatmap(bids, asks)

        elif symbol == 'BTCUSDT':
            df_btcusdt = df_btcusdt.append({'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': timestamp}, ignore_index=True)
            
        elif symbol == 'ETHBTC':
            df_ethbtc = df_ethbtc.append({'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': timestamp}, ignore_index=True)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("### connection closed ###")
    df_ethusdt.to_csv('ethusdt2.csv', index=False)
    df_btcusdt.to_csv('btcusdt2.csv', index=False)
    df_ethbtc.to_csv('ethbtc2.csv', index=False)

def on_open(ws):
    print("### connection opened ###")
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": [
            "btcusdt@depth",
            "ethusdt@depth",
            "ethbtc@depth"
        ],
        "id": 1
    }))

if __name__ == "__main__":
    ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()


import pandas as pd
import ast
import matplotlib.pyplot as plt
import math

# Read data from csv files
df_ethusdt = pd.read_csv('ethusdt.csv')
df_btcusdt = pd.read_csv('btcusdt.csv')
df_ethbtc = pd.read_csv('ethbtc.csv')

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except ValueError as e:
        print(f"Failed to evaluate string: {s}, error: {e}")
        return None

df_ethusdt['asks'] = df_ethusdt['asks'].apply(safe_literal_eval)
df_ethusdt['bids'] = df_ethusdt['bids'].apply(safe_literal_eval)
df_btcusdt['asks'] = df_btcusdt['asks'].apply(safe_literal_eval)
df_btcusdt['bids'] = df_btcusdt['bids'].apply(safe_literal_eval)
df_ethbtc['asks'] = df_ethbtc['asks'].apply(safe_literal_eval)
df_ethbtc['bids'] = df_ethbtc['bids'].apply(safe_literal_eval)

# Initial capital
capital = 100000  # USDT
capital_without_TC = 100000  # USDT without transaction costs
capital_without_TC_and_vol = 100000  # USDT without transaction costs and volatility

# Initialize lists to store capital evolution
capital_evolution = [capital]
capital_evolution_without_TC = [capital_without_TC]
capital_evolution_without_TC_and_vol = [capital_without_TC_and_vol]

# Initialize list to store cumulative transaction costs
transaction_costs_cumulative = [0]

# Calculate slippage based on annual volatility of 65%
slippage = 0.65 * math.sqrt(1 / 252)

# Percentage of capital to invest in each opportunity
investment_fraction = 0.01  # 1%

# Counter for arbitrage opportunities
arbitrage_counter = 0

def execute_arbitrage_opportunity(usdt, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, with_TC=True, with_vol=True,fee=0.001):
    fee = 0.001 if with_TC else 0
    vol = slippage if with_vol else 0
    
    eth_usdt_ask *= (1 + vol)
    btc_usdt_bid *= (1 - vol)
    eth_btc_bid *= (1 - vol)

    eth = usdt / eth_usdt_ask * (1 - fee)
    btc = eth * eth_btc_bid * (1 - fee)
    balance_usdt = btc * btc_usdt_bid * (1 - fee)

    return balance_usdt

def calculate_transaction_cost(usdt, eth_usdt_ask, btc_usdt_bid, eth_btc_bid):
    eth_usdt_ask *= (1 + slippage)
    btc_usdt_bid *= (1 - slippage)
    eth_btc_bid *= (1 - slippage)
    
    cost = 0
    eth = usdt / eth_usdt_ask
    cost += usdt * 0.001  # Fee for ETH/USDT
    
    btc = eth * eth_btc_bid
    cost += (eth * eth_btc_bid) * 0.001  # Fee for ETH/BTC
    
    balance_usdt = btc * btc_usdt_bid
    cost += (btc * btc_usdt_bid) * 0.001  # Fee for BTC/USDT
    
    return cost

# Iterate over the minimum length of the dataframes
min_length = min(len(df_ethusdt), len(df_btcusdt), len(df_ethbtc))

for i in range(min_length):
    eth_usdt_ask = float(df_ethusdt.iloc[i]['asks'][0][0]) if len(df_ethusdt.iloc[i]['asks']) > 0 else 0
    btc_usdt_bid = float(df_btcusdt.iloc[i]['bids'][0][0]) if len(df_btcusdt.iloc[i]['bids']) > 0 else 0
    eth_btc_bid = float(df_ethbtc.iloc[i]['bids'][0][0]) if len(df_ethbtc.iloc[i]['bids']) > 0 else 0
    
    if eth_usdt_ask > 0 and btc_usdt_bid > 0 and eth_btc_bid > 0:
        arbitrage_counter += 1
        
        investment = capital * investment_fraction
        investment_without_TC = capital_without_TC * investment_fraction
        investment_without_TC_and_vol = capital_without_TC_and_vol * investment_fraction

        new_balance = execute_arbitrage_opportunity(investment, eth_usdt_ask, btc_usdt_bid, eth_btc_bid)
        new_balance_without_TC = execute_arbitrage_opportunity(investment_without_TC, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, with_TC=False)
        new_balance_without_TC_and_vol = execute_arbitrage_opportunity(investment_without_TC_and_vol, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, with_TC=False, with_vol=False)

        capital = capital - investment + new_balance
        capital_without_TC = capital_without_TC - investment_without_TC + new_balance_without_TC
        capital_without_TC_and_vol = capital_without_TC_and_vol - investment_without_TC_and_vol + new_balance_without_TC_and_vol

        capital_evolution.append(capital)
        capital_evolution_without_TC.append(capital_without_TC)
        capital_evolution_without_TC_and_vol.append(capital_without_TC_and_vol)
        
        cost_this_step = calculate_transaction_cost(investment, eth_usdt_ask, btc_usdt_bid, eth_btc_bid)
        transaction_costs_cumulative.append(transaction_costs_cumulative[-1] + cost_this_step)

# Results
print(f'Final capital: {capital} USDT')
print(f'Final capital (without transaction costs): {capital_without_TC} USDT')
print(f'Final capital (without transaction costs and slippage): {capital_without_TC_and_vol} USDT')
print(f'Number of arbitrage opportunities: {arbitrage_counter}')

# Plotting capital evolution
plt.figure()
plt.plot([(x / 100000) * 100 for x in capital_evolution], label='With TC and Slippage')
plt.plot([(x / 100000) * 100 for x in capital_evolution_without_TC], label='Without TC')
plt.plot([(x / 100000) * 100 for x in capital_evolution_without_TC_and_vol], label='Without TC and Slippage')
plt.xlabel('Step')
plt.ylabel('Capital (%)')
plt.legend()
plt.show()

# Plotting transaction costs
plt.figure()
plt.plot(transaction_costs_cumulative, label='Cumulative Transaction Costs')
plt.xlabel('Step')
plt.ylabel('Cumulative Costs (USDT)')
plt.legend()
plt.show()

from scipy import stats
import statsmodels.api as sm
import numpy as np

# Test de Student (t-test) pour évaluer si les coûts de transaction sont statistiquement significatifs ou non
print("\nStudent's t-test to evaluate if transaction costs are statistically significant:")
capital_evolution_array = np.array(capital_evolution)
capital_evolution_without_TC_array = np.array(capital_evolution_without_TC)

# Calculate the pairwise differences between capital evolution with and without transaction costs
pairwise_differences = capital_evolution_array - capital_evolution_without_TC_array

# Student's t-test on the pairwise differences
t_stat_diff, p_val_diff = stats.ttest_1samp(pairwise_differences, 0)
print(f"\nStudent's t-test on pairwise differences:")
print(f'T-statistic: {t_stat_diff}')
print(f'P-value: {p_val_diff}')


# Analyse de régression pour évaluer l'impact des coûts de transaction sur le capital
print("\nRegression Analysis:")
X = capital_evolution_without_TC_array  # Variable indépendante
y = capital_evolution_array  # Variable dépendante
X = sm.add_constant(X)  # Ajoute une constante (intercept) au modèle
model = sm.OLS(y, X).fit()
print(model.summary())

# Calculate the pairwise differences between capital evolution with and without slippage
pairwise_differences_slippage = capital_evolution_without_TC_array - capital_evolution_without_TC_and_vol

# Student's t-test on the pairwise differences
t_stat_diff_slippage, p_val_diff_slippage = stats.ttest_1samp(pairwise_differences_slippage, 0)
print(f"\nStudent's t-test on pairwise differences due to slippage:")
print(f'T-statistic: {t_stat_diff_slippage}')
print(f'P-value: {p_val_diff_slippage}')


# Fonction pour calculer le capital final à un taux de coût de transaction donné
def execute_arbitrage_opportunity2(usdt, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, transaction_cost_rate):
    eth_usdt_ask *= (1 + slippage)
    btc_usdt_bid *= (1 - slippage)
    eth_btc_bid *= (1 - slippage)
    
    eth = usdt / eth_usdt_ask * (1 - transaction_cost_rate)
    btc = eth * eth_btc_bid * (1 - transaction_cost_rate)
    balance_usdt = btc * btc_usdt_bid * (1 - transaction_cost_rate)

    return balance_usdt

# Calculer le capital final pour différents taux de coûts de transaction
import numpy as np
transaction_cost_rates = np.linspace(0.001, 0.01, 10)  # de 0,1% à 1%

def compute_final_capital2(transaction_cost_rate, min_length, investment_fraction):
    capital = 100000  # Capital initial
    for i in range(min_length):
        eth_usdt_ask = float(df_ethusdt.iloc[i]['asks'][0][0]) if len(df_ethusdt.iloc[i]['asks']) > 0 else 0
        btc_usdt_bid = float(df_btcusdt.iloc[i]['bids'][0][0]) if len(df_btcusdt.iloc[i]['bids']) > 0 else 0
        eth_btc_bid = float(df_ethbtc.iloc[i]['bids'][0][0]) if len(df_ethbtc.iloc[i]['bids']) > 0 else 0
        
        if eth_usdt_ask > 0 and btc_usdt_bid > 0 and eth_btc_bid > 0:
            investment = capital * investment_fraction
            new_balance = execute_arbitrage_opportunity2(investment, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, transaction_cost_rate)
            capital = capital - investment + new_balance
    return capital

final_capitals2 = [compute_final_capital2(rate, min_length, investment_fraction) for rate in transaction_cost_rates]

# Tracer le capital final en fonction du taux de coût de transaction
plt.figure()
plt.plot(transaction_cost_rates * 100, [(x / 100000 - 1) * 100 for x in final_capitals2])
plt.xlabel('Taux de coût de transaction (%)')
plt.ylabel('PnL (%)')
plt.title('PnL en fonction du taux de coût de transaction')
plt.show()

# Fonction pour calculer le capital final à un taux de slippage donné
def execute_arbitrage_opportunity3(usdt, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, slippage_rate):
    eth_usdt_ask *= (1 + slippage_rate)
    btc_usdt_bid *= (1 - slippage_rate)
    eth_btc_bid *= (1 - slippage_rate)
    
    eth = usdt / eth_usdt_ask * (1 - 0.001)
    btc = eth * eth_btc_bid * (1 - 0.001)
    balance_usdt = btc * btc_usdt_bid * (1 - 0.001)

    return balance_usdt

# Calculer le capital final pour différents taux de slippage
slippage_rates = np.linspace(0.001, 0.06, 60)  # de 0.1% à 6%

def compute_final_capital3(slippage_rate, min_length, investment_fraction):
    capital = 100000  # Capital initial
    for i in range(min_length):
        eth_usdt_ask = float(df_ethusdt.iloc[i]['asks'][0][0]) if len(df_ethusdt.iloc[i]['asks']) > 0 else 0
        btc_usdt_bid = float(df_btcusdt.iloc[i]['bids'][0][0]) if len(df_btcusdt.iloc[i]['bids']) > 0 else 0
        eth_btc_bid = float(df_ethbtc.iloc[i]['bids'][0][0]) if len(df_ethbtc.iloc[i]['bids']) > 0 else 0
        
        if eth_usdt_ask > 0 and btc_usdt_bid > 0 and eth_btc_bid > 0:
            investment = capital * investment_fraction
            new_balance = execute_arbitrage_opportunity3(investment, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, slippage_rate)
            capital = capital - investment + new_balance
    return capital

final_capitals3 = [compute_final_capital3(rate, min_length, investment_fraction) for rate in slippage_rates]

# Tracer le capital final en fonction du taux de slippage
plt.figure()
plt.plot(slippage_rates * 100, [(x / 100000 - 1) * 100 for x in final_capitals3])
plt.scatter([4.1], [(compute_final_capital3(0.041, min_length, investment_fraction) / 100000 - 1) * 100], color='red')  # Point rouge à 4.1%
plt.xlabel('Taux de slippage (%)')
plt.ylabel('PnL (%)')
plt.title('PnL en fonction du taux de slippage')
plt.show()


