import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_plots():
    print("Generating EDA plots...")
    sns.set_theme(style="whitegrid")
    os.makedirs("images", exist_ok=True)

    # 1. Load raw data
    df = pd.read_csv("data/train.csv")
    df['date'] = pd.to_datetime(df['date'])

    # 2. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['store', 'item', 'price', 'sales']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved correlation_heatmap.png")

    # 3. Sales trend
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales['date'], daily_sales['sales'], color='royalblue', alpha=0.8)
    plt.title("Total Daily Sales Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Units")
    plt.tight_layout()
    plt.savefig("images/sales_trend.png", dpi=150)
    plt.close()
    print("Saved sales_trend.png")

    # 4. Monthly sales
    df['month'] = df['date'].dt.month
    monthly_sales = df.groupby('month')['sales'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=monthly_sales, x='month', y='sales', palette='Blues_r')
    plt.title("Average Sales by Month (Seasonality)")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("images/monthly_sales.png", dpi=150)
    plt.close()
    print("Saved monthly_sales.png")

    # 5. Store sales
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='store', y='sales', palette='Set2')
    plt.title("Sales Distribution by Store")
    plt.xlabel("Store ID")
    plt.ylabel("Sales Units")
    plt.tight_layout()
    plt.savefig("images/store_sales.png", dpi=150)
    plt.close()
    print("Saved store_sales.png")

    # 6. Item sales
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='item', y='sales', palette='Set3')
    plt.title("Sales Distribution by Item")
    plt.xlabel("Item ID")
    plt.ylabel("Sales Units")
    plt.tight_layout()
    plt.savefig("images/item_sales.png", dpi=150)
    plt.close()
    print("Saved item_sales.png")

    # 7. Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['sales'], bins=30, kde=True, ax=axes[0], color='indigo')
    axes[0].set_title("Distribution of Sales")
    axes[0].set_xlabel("Sales Units")
    sns.histplot(df['price'], bins=30, kde=True, ax=axes[1], color='teal')
    axes[1].set_title("Distribution of Prices")
    axes[1].set_xlabel("Price ($)")
    plt.tight_layout()
    plt.savefig("images/distributions.png", dpi=150)
    plt.close()
    print("Saved distributions.png")

    # 8. Demand curve for Item 1
    item_df = df[df['item'] == 1]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=item_df, x='price', y='sales', alpha=0.5, color='darkorange')
    plt.title("Demand Curve for Item 1: Sales vs Price")
    plt.xlabel("Price ($)")
    plt.ylabel("Sales Units")
    plt.tight_layout()
    plt.savefig("images/demand_curve_item1.png", dpi=150)
    plt.close()
    print("Saved demand_curve_item1.png")
    
    print("All EDA plots generated successfully!")

if __name__ == "__main__":
    generate_eda_plots()
