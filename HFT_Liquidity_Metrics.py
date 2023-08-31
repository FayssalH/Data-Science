# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:43:43 2023

@author: Fayssal
"""


import pandas as pd
import ast
import matplotlib.pyplot as plt
import math

# Read data from csv files
df_ethusdt = pd.read_csv('ethusdt2.csv')
df_btcusdt = pd.read_csv('btcusdt2.csv')
df_ethbtc = pd.read_csv('ethbtc2.csv')

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


def calculate_liquidity(df):
    def metrics(row):
        if len(row['bids']) > 0 and len(row['asks']) > 0:
            best_bid = float(row['bids'][0][0])
            best_ask = float(row['asks'][0][0])

            if best_ask <= best_bid:  # Filtre pour s'assurer que ask > bid
                return pd.Series([None, None, None], index=['Mid_Price', 'Quoted_Spread', 'First_Layer_Liquidity'])

            best_bid_volume = float(row['bids'][0][1])
            best_ask_volume = float(row['asks'][0][1])
            
            mid_price = (best_bid + best_ask) / 2
            quoted_spread = best_ask - best_bid
            first_layer_liquidity = best_bid_volume + best_ask_volume
            
            return pd.Series([mid_price, quoted_spread, first_layer_liquidity], index=['Mid_Price', 'Quoted_Spread', 'First_Layer_Liquidity'])
        else:
            return pd.Series([None, None, None], index=['Mid_Price', 'Quoted_Spread', 'First_Layer_Liquidity'])

    df[['Mid_Price', 'Quoted_Spread', 'First_Layer_Liquidity']] = df.apply(metrics, axis=1)
    return df

df_ethusdt = calculate_liquidity(df_ethusdt)
df_btcusdt = calculate_liquidity(df_btcusdt)
df_ethbtc = calculate_liquidity(df_ethbtc)

df_ethusdt.to_csv('ethusdt2_with_liquidity_metrics.csv', index=False)
df_btcusdt.to_csv('btcusdt2_with_liquidity_metrics.csv', index=False)
df_ethbtc.to_csv('ethbtc2_with_liquidity_metrics.csv', index=False)

def plot_liquidity_metrics(df, title, x_limits=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))

    ax[0].hist(df['Quoted_Spread'].dropna(), bins=30, alpha=0.7, color='red', label='Quoted Spread')
    ax[0].set_title('Distribution of Quoted Spread')
    ax[0].set_xlabel('Spread')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()

    # Increase the number of bins and/or set x-axis limits
    ax[1].hist(df['First_Layer_Liquidity'].dropna(), bins=100, alpha=0.7, color='green', label='First Layer Liquidity')  
    ax[1].set_title('Distribution of First Layer Liquidity')
    ax[1].set_xlabel('Volume')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    
    if x_limits:
        ax[1].set_xlim(x_limits)

    plt.suptitle(title)
    plt.show()



# Plot liquidity metrics for each pair
plot_liquidity_metrics(df_ethusdt, 'ETHUSDT Liquidity Metrics', x_limits=(50, 200))
plot_liquidity_metrics(df_btcusdt, 'BTCUSDT Liquidity Metrics', x_limits=(0, 50))
plot_liquidity_metrics(df_ethbtc, 'ETHBTC Liquidity Metrics', x_limits=(0, 400))
