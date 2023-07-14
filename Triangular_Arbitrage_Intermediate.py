# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:47:55 2023

@author: Fayssal
"""

import websocket
import json
import pandas as pd

# Create empty DataFrame to store incoming data
df_ethusdt = pd.DataFrame(columns=['symbol', 'bids', 'asks', 'timestamp'])
df_btcusdt = pd.DataFrame(columns=['symbol', 'bids', 'asks', 'timestamp'])
df_ethbtc = pd.DataFrame(columns=['symbol', 'bids', 'asks', 'timestamp'])

def on_message(ws, message):
    global df_ethusdt
    global df_btcusdt
    global df_ethbtc

    data = json.loads(message)

    # Check if the message is an orderbook update
    if "e" in data and data["e"] == "depthUpdate":
        symbol = data['s']
        bids = data['b']  # [price, quantity] 
        asks = data['a']  # [price, quantity]
        timestamp = data['E']

        # Update specific dataframes
        if symbol == 'ETHUSDT':
            df_ethusdt = df_ethusdt.append({'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': timestamp}, ignore_index=True)
        elif symbol == 'BTCUSDT':
            df_btcusdt = df_btcusdt.append({'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': timestamp}, ignore_index=True)
        elif symbol == 'ETHBTC':
            df_ethbtc = df_ethbtc.append({'symbol': symbol, 'bids': bids, 'asks': asks, 'timestamp': timestamp}, ignore_index=True)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("### connection closed ###")

    # Save data to csv when connection is closed
    df_ethusdt.to_csv('ethusdt.csv', index=False)
    df_btcusdt.to_csv('btcusdt.csv', index=False)
    df_ethbtc.to_csv('ethbtc.csv', index=False)
    print("Saved data to csv files.")

def on_open(ws):
    print("### connection opened ###")

    # Binance requires you to send a subscribe message to start receiving updates
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

# Read data from csv files
df_ethusdt = pd.read_csv('ethusdt.csv')
df_btcusdt = pd.read_csv('btcusdt.csv')
df_ethbtc = pd.read_csv('ethbtc.csv')

# Convert 'asks' and 'bids' columns from string format to actual lists
df_ethusdt['asks'] = df_ethusdt['asks'].apply(ast.literal_eval)
df_ethusdt['bids'] = df_ethusdt['bids'].apply(ast.literal_eval)
df_btcusdt['asks'] = df_btcusdt['asks'].apply(ast.literal_eval)
df_btcusdt['bids'] = df_btcusdt['bids'].apply(ast.literal_eval)
df_ethbtc['asks'] = df_ethbtc['asks'].apply(ast.literal_eval)
df_ethbtc['bids'] = df_ethbtc['bids'].apply(ast.literal_eval)

# Initial capital
capital = 100000  # USDT
capital_without_TC = 100000  # USDT without transaction costs

# Set initial time of opportunity
opportunity_start_time = None
opportunity_start_time_without_TC = None

# Initialize lists to store durations of arbitrage opportunities
arbitrage_durations = []
arbitrage_durations_without_TC = []

def execute_arbitrage_opportunity(usdt, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, with_TC=True):
    fee = 0.001 if with_TC else 0  # Transaction fee (0.1%) if with transaction costs

    # Initial balance (in USDT)
    balance_usdt = usdt

    # Buy ETH with USDT
    eth = balance_usdt / eth_usdt_ask * (1 - fee)
    balance_usdt = 0

    # Sell ETH for BTC
    btc = eth * eth_btc_bid * (1 - fee)
    eth = 0

    # Sell BTC for USDT
    balance_usdt = btc * btc_usdt_bid * (1 - fee)
    btc = 0

    return balance_usdt

# Iterate over the minimum length of the dataframes
min_length = min(len(df_ethusdt), len(df_btcusdt), len(df_ethbtc))

for i in range(min_length):
    if len(df_ethusdt.iloc[i]['asks']) > 0 and len(df_btcusdt.iloc[i]['bids']) > 0 and len(df_ethbtc.iloc[i]['bids']) > 0:
        eth_usdt_ask = float(df_ethusdt.iloc[i]['asks'][0][0])
        btc_usdt_bid = float(df_btcusdt.iloc[i]['bids'][0][0])
        eth_btc_bid = float(df_ethbtc.iloc[i]['bids'][0][0])

        new_balance = execute_arbitrage_opportunity(capital, eth_usdt_ask, btc_usdt_bid, eth_btc_bid)
        new_balance_without_TC = execute_arbitrage_opportunity(capital_without_TC, eth_usdt_ask, btc_usdt_bid, eth_btc_bid, with_TC=False)

        # If there is an arbitrage opportunity
        if new_balance > capital:
            if opportunity_start_time is None:
                opportunity_start_time = df_ethusdt.iloc[i]['timestamp']  # Opportunity starts
                print('Arbitrage opportunity started')
            capital = new_balance  # Update capital with new balance

        elif opportunity_start_time is not None:
            opportunity_end_time = df_ethusdt.iloc[i]['timestamp']  # Opportunity ends
            duration = (opportunity_end_time - opportunity_start_time) / 1000  # in seconds
            arbitrage_durations.append(duration)  # Store duration
            print(f'Arbitrage opportunity (with transaction costs) ended. Duration: {duration:.3f} s')
            opportunity_start_time = None  # Reset opportunity start time

        # If there is an arbitrage opportunity without transaction costs
        if new_balance_without_TC > capital_without_TC:
            if opportunity_start_time_without_TC is None:
                opportunity_start_time_without_TC = df_ethusdt.iloc[i]['timestamp']  # Opportunity (without transaction costs) starts
                print('Arbitrage opportunity (without transaction costs) started')
            capital_without_TC = new_balance_without_TC  # Update capital with new balance

        elif opportunity_start_time_without_TC is not None:
            opportunity_end_time_without_TC = df_ethusdt.iloc[i]['timestamp']  # Opportunity ends
            duration_without_TC = (opportunity_end_time_without_TC - opportunity_start_time_without_TC) / 1000  # in seconds
            arbitrage_durations_without_TC.append(duration_without_TC)  # Store duration
            print(f'Arbitrage opportunity (without transaction costs) ended. Duration: {duration_without_TC:.3f} s')
            opportunity_start_time_without_TC = None  # Reset opportunity start time

print(f'Final capital (with transaction costs): {capital} USDT')
print(f'Final capital (without transaction costs): {capital_without_TC} USDT')

# Plot arbitrage durations
plt.figure(figsize=(10, 5))
plt.plot(arbitrage_durations, label='With transaction costs')
plt.plot(arbitrage_durations_without_TC, label='Without transaction costs')
plt.xlabel('Arbitrage opportunity number')
plt.ylabel('Duration (s)')
plt.legend()
plt.show()





