import os
import numpy as np
import pandas as pd

def generate_synthetic_data(seed=42):
    np.random.seed(seed)
    
    # Date range: 4 years from 2022 to 2025
    dates = pd.date_range(start="2022-01-01", end="2025-12-31", freq="D")
    num_days = len(dates)
    
    # Config
    stores = [1, 2, 3]
    items = [1, 2, 3, 4, 5]
    
    # Store multipliers
    store_multipliers = {1: 1.0, 2: 1.4, 3: 0.8}
    
    # Item configuration: base demand, base price, and price elasticity
    item_config = {
        1: {"base_demand": 50, "base_price": 10.0, "elasticity": 1.2},
        2: {"base_demand": 120, "base_price": 25.0, "elasticity": 1.5},
        3: {"base_demand": 30, "base_price": 5.0, "elasticity": 0.8},
        4: {"base_demand": 80, "base_price": 15.0, "elasticity": 1.1},
        5: {"base_demand": 150, "base_price": 40.0, "elasticity": 1.7}
    }
    
    data = []
    
    for store in stores:
        store_mult = store_multipliers[store]
        for item in items:
            config = item_config[item]
            base_demand = config["base_demand"]
            base_price = config["base_price"]
            elasticity = config["elasticity"]
            
            # Generate price array with some random discounts and price changes
            # Base price has a slow drift, and occasional promotional discount dips
            prices = []
            for t in range(num_days):
                # Drift: very slow sine wave + small noise
                drift = 0.05 * base_price * np.sin(2 * np.pi * t / 365)
                noise = np.random.normal(0, 0.02 * base_price)
                price = base_price + drift + noise
                
                # Promotions: 5% chance of a discount (10%, 20%, or 30%)
                if np.random.rand() < 0.05:
                    discount = np.random.choice([0.1, 0.2, 0.3])
                    price = price * (1.0 - discount)
                
                # Ensure price doesn't go below a reasonable threshold
                price = max(0.5 * base_price, round(price, 2))
                prices.append(price)
            
            # Generate sales based on trend, seasonality, price, and noise
            for t, date in enumerate(dates):
                day_of_year = date.dayofyear
                day_of_week = date.dayofweek # 0 is Monday, 6 is Sunday
                
                # 1. Trend: slow upward growth
                trend = t * 0.01
                
                # 2. Weekly seasonality: weekends have higher demand
                # Monday-Thursday: 0, Friday: +0.15, Saturday: +0.3, Sunday: +0.2
                weekly_factor = 1.0
                if day_of_week == 4: # Friday
                    weekly_factor = 1.15
                elif day_of_week == 5: # Saturday
                    weekly_factor = 1.35
                elif day_of_week == 6: # Sunday
                    weekly_factor = 1.25
                
                # 3. Yearly seasonality: sine/cosine waves
                # Peak in summer (July, day ~200) and holiday season (December, day ~350)
                yearly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365) + 0.10 * np.cos(4 * np.pi * day_of_year / 365)
                
                # 4. Price Elasticity impact:
                # Sales decrease exponentially as price rises relative to base price
                current_price = prices[t]
                price_ratio = current_price / base_price
                # Demand curve: Demand(P) = Demand_Base * (P/P_Base)^(-elasticity)
                price_factor = (price_ratio) ** (-elasticity)
                
                # Combine factors
                expected_sales = (base_demand + trend) * weekly_factor * yearly_factor * price_factor * store_mult
                
                # Add random noise (scaled with expected sales to prevent negative values)
                noise = np.random.normal(0, 0.08 * expected_sales + 2)
                sales = max(0, int(round(expected_sales + noise)))
                
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store": store,
                    "item": item,
                    "price": current_price,
                    "sales": sales
                })
                
    df = pd.DataFrame(data)
    
    # Save datasets
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/train.csv", index=False)
    print(f"Generated train.csv with {len(df)} rows.")
    
    # Generate sample_data.csv (a smaller subset, e.g., the last 150 rows)
    sample_df = df.tail(150)
    sample_df.to_csv("data/sample_data.csv", index=False)
    print(f"Generated sample_data.csv with {len(sample_df)} rows.")

if __name__ == "__main__":
    generate_synthetic_data()
