import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook {filename}")

def build_eda_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01. Exploratory Data Analysis (EDA)\n",
                "\n",
                "This notebook performs exploratory analysis on our retail sales dataset. We will analyze missing values, correlations, trends, monthly/weekly seasonality, store-wise and item-wise distributions, and price elasticity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "os.makedirs(\"../images\", exist_ok=True)\n",
                "\n",
                "# Load raw data\n",
                "df = pd.read_csv(\"../data/train.csv\")\n",
                "df['date'] = pd.to_datetime(df['date'])\n",
                "print(f\"Dataset shape: {df.shape}\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Missing Value Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Missing values check:\")\n",
                "print(df.isnull().sum())\n",
                "\n",
                "print(\"\\nData Types:\")\n",
                "print(df.dtypes)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Correlation Analysis\n",
                "\n",
                "Let's see how numeric columns relate to sales, including our generated `price` column."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 6))\n",
                "corr = df[['store', 'item', 'price', 'sales']].corr()\n",
                "sns.heatmap(corr, annot=True, cmap=\"coolwarm\", fmt=\".3f\", vmin=-1, vmax=1)\n",
                "plt.title(\"Correlation Heatmap\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/correlation_heatmap.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Overall Sales Trend Over Time"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "daily_sales = df.groupby('date')['sales'].sum().reset_index()\n",
                "plt.figure(figsize=(12, 6))\n",
                "plt.plot(daily_sales['date'], daily_sales['sales'], color='royalblue', alpha=0.8)\n",
                "plt.title(\"Total Daily Sales Trend Over Time\")\n",
                "plt.xlabel(\"Date\")\n",
                "plt.ylabel(\"Sales Units\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/sales_trend.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Monthly Sales Analysis\n",
                "\n",
                "Let's extract the month and check seasonality."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['month'] = df['date'].dt.month\n",
                "monthly_sales = df.groupby('month')['sales'].mean().reset_index()\n",
                "\n",
                "plt.figure(figsize=(8, 5))\n",
                "sns.barplot(data=monthly_sales, x='month', y='sales', palette='Blues_r')\n",
                "plt.title(\"Average Sales by Month (Seasonality check)\")\n",
                "plt.xlabel(\"Month\")\n",
                "plt.ylabel(\"Average Sales\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/monthly_sales.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Store-wise Sales Performance"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 5))\n",
                "sns.boxplot(data=df, x='store', y='sales', palette='Set2')\n",
                "plt.title(\"Sales Distribution by Store\")\n",
                "plt.xlabel(\"Store ID\")\n",
                "plt.ylabel(\"Sales Units\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/store_sales.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Item-wise Sales Performance"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 5))\n",
                "sns.boxplot(data=df, x='item', y='sales', palette='Set3')\n",
                "plt.title(\"Sales Distribution by Item\")\n",
                "plt.xlabel(\"Item ID\")\n",
                "plt.ylabel(\"Sales Units\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/item_sales.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Distribution of Sales and Prices"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "sns.histplot(df['sales'], bins=30, kde=True, ax=axes[0], color='indigo')\n",
                "axes[0].set_title(\"Distribution of Sales\")\n",
                "axes[0].set_xlabel(\"Sales Units\")\n",
                "\n",
                "sns.histplot(df['price'], bins=30, kde=True, ax=axes[1], color='teal')\n",
                "axes[1].set_title(\"Distribution of Prices\")\n",
                "axes[1].set_xlabel(\"Price ($)\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/distributions.png\", dpi=150)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Price Sensitivity / Elasticity Analysis (Demand Curve)\n",
                "\n",
                "Let's see how sales volume is affected by prices for a specific item (e.g. Item 1)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "item_id = 1\n",
                "item_df = df[df['item'] == item_id]\n",
                "\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.scatterplot(data=item_df, x='price', y='sales', alpha=0.5, color='darkorange')\n",
                "plt.title(f\"Demand Curve for Item {item_id}: Sales vs Price\")\n",
                "plt.xlabel(\"Price ($)\")\n",
                "plt.ylabel(\"Sales Units\")\n",
                "plt.tight_layout()\n",
                "plt.savefig(\"../images/demand_curve_item1.png\", dpi=150)\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook("notebooks/01_EDA.ipynb", cells)

def build_preprocessing_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02. Preprocessing & Feature Engineering\n",
                "\n",
                "This notebook details the preprocessing and feature engineering steps. In time series forecasting, we extract temporal features (Year, Month, etc.) and create historical features like lags and rolling statistics to capture autocorrelation, ensuring we group by `store` and `item` to avoid leakage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "# Load raw data\n",
                "df = pd.read_csv(\"../data/train.csv\")\n",
                "df['date'] = pd.to_datetime(df['date'])\n",
                "df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Extract Time Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def extract_time_features(data):\n",
                "    df = data.copy()\n",
                "    df['year'] = df['date'].dt.year\n",
                "    df['month'] = df['date'].dt.month\n",
                "    df['week'] = df['date'].dt.isocalendar().week.astype(int)\n",
                "    df['day'] = df['date'].dt.day\n",
                "    df['dayofweek'] = df['date'].dt.dayofweek\n",
                "    df['quarter'] = df['date'].dt.quarter\n",
                "    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)\n",
                "    return df\n",
                "\n",
                "df_time = extract_time_features(df)\n",
                "df_time.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Generate Lag Features\n",
                "\n",
                "Lag features help our model learn past sales behavior. We will create lags of 7, 14, and 30 days."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def add_lags(df, lags=[7, 14, 30]):\n",
                "    for lag in lags:\n",
                "        df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)\n",
                "    return df\n",
                "\n",
                "df_lags = add_lags(df_time)\n",
                "df_lags[['date', 'store', 'item', 'sales', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30']].head(10)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Generate Rolling Mean Features\n",
                "\n",
                "Rolling averages smooth out noise and capture recent trends. We compute 7, 14, and 30 days rolling means."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def add_rolling_means(df, windows=[7, 14, 30]):\n",
                "    for window in windows:\n",
                "        # In time series forecasting, we shift by 1 before calculating rolling statistics\n",
                "        # to avoid data leakage (knowing today's sales to predict today's sales)\n",
                "        df[f'sales_roll_mean_{window}'] = df.groupby(['store', 'item'])['sales'] \\\n",
                "                                              .shift(1) \\\n",
                "                                              .transform(lambda x: x.rolling(window=window).mean())\n",
                "    return df\n",
                "\n",
                "df_rolling = add_rolling_means(df_lags)\n",
                "df_rolling[['date', 'store', 'item', 'sales', 'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30']].head(12)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Handling NaN Values\n",
                "\n",
                "Shifting and rolling create `NaN` values at the beginning of each store-item timeline. We will drop these rows."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f\"Shape before dropping NaNs: {df_rolling.shape}\")\n",
                "df_final = df_rolling.dropna().reset_index(drop=True)\n",
                "print(f\"Shape after dropping NaNs: {df_final.shape}\")\n",
                "df_final.head()"
            ]
        }
    ]
    create_notebook("notebooks/02_Preprocessing.ipynb", cells)

def build_training_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 03. Model Training & Comparison\n",
                "\n",
                "This notebook trains three different models to forecast sales: Linear Regression, Random Forest, and XGBoost. We evaluate them on a time-based train-test split using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score, then save the best model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import joblib\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from xgboost import XGBRegressor\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "os.makedirs(\"../models\", exist_ok=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load and Prepare Engineered Features\n",
                "\n",
                "We will apply the feature engineering pipeline from notebook 02."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Read data\n",
                "df = pd.read_csv(\"../data/train.csv\")\n",
                "df['date'] = pd.to_datetime(df['date'])\n",
                "df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)\n",
                "\n",
                "# Feature engineering\n",
                "df['year'] = df['date'].dt.year\n",
                "df['month'] = df['date'].dt.month\n",
                "df['week'] = df['date'].dt.isocalendar().week.astype(int)\n",
                "df['day'] = df['date'].dt.day\n",
                "df['dayofweek'] = df['date'].dt.dayofweek\n",
                "df['quarter'] = df['date'].dt.quarter\n",
                "df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)\n",
                "\n",
                "# Lags & rolling means\n",
                "for lag in [7, 14, 30]:\n",
                "    df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)\n",
                "\n",
                "for window in [7, 14, 30]:\n",
                "    df[f'sales_roll_mean_{window}'] = df.groupby(['store', 'item'])['sales'] \\\n",
                "                                          .shift(1) \\\n",
                "                                          .transform(lambda x: x.rolling(window=window).mean())\n",
                "\n",
                "# Drop NaNs\n",
                "df = df.dropna().reset_index(drop=True)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Time-Based Train-Test Split\n",
                "\n",
                "Using a random train-test split for time series would cause look-ahead leakage. Instead, we split chronologically: \n",
                "- **Train**: 2022-01-01 to 2025-06-01\n",
                "- **Test**: 2025-06-01 to 2025-12-31"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "split_date = pd.to_datetime(\"2025-06-01\")\n",
                "\n",
                "train_df = df[df['date'] < split_date]\n",
                "test_df = df[df['date'] >= split_date]\n",
                "\n",
                "features = [\n",
                "    'store', 'item', 'price',\n",
                "    'year', 'month', 'week', 'day', 'dayofweek', 'quarter', 'is_weekend',\n",
                "    'sales_lag_7', 'sales_lag_14', 'sales_lag_30',\n",
                "    'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_30'\n",
                "]\n",
                "target = 'sales'\n",
                "\n",
                "X_train, y_train = train_df[features], train_df[target]\n",
                "X_test, y_test = test_df[features], test_df[target]\n",
                "\n",
                "print(f\"Train shape: {X_train.shape}\")\n",
                "print(f\"Test shape: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Scale Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "# Save the scaler\n",
                "joblib.dump(scaler, \"../models/scaler.pkl\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Train Models & Evaluate"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = {\n",
                "    \"Linear Regression\": LinearRegression(),\n",
                "    \"Random Forest\": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),\n",
                "    \"XGBoost\": XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1)\n",
                "}\n",
                "\n",
                "results = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    print(f\"Training {name}...\")\n",
                "    model.fit(X_train_scaled, y_train)\n",
                "    preds = model.predict(X_test_scaled)\n",
                "    \n",
                "    # Metric evaluations\n",
                "    mae = mean_absolute_error(y_test, preds)\n",
                "    rmse = np.sqrt(mean_squared_error(y_test, preds))\n",
                "    r2 = r2_score(y_test, preds)\n",
                "    \n",
                "    results[name] = {\n",
                "        \"MAE\": mae,\n",
                "        \"RMSE\": rmse,\n",
                "        \"R2 Score\": r2\n",
                "    }\n",
                "\n",
                "results_df = pd.DataFrame(results).T\n",
                "results_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Identify Best Model and Save\n",
                "\n",
                "Save the best model based on MAE/R2. Generally, XGBoost or Random Forest outperforms Linear Regression."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Find best model based on lowest MAE\n",
                "best_model_name = results_df['MAE'].idxmin()\n",
                "best_model = models[best_model_name]\n",
                "print(f\"Best model is: {best_model_name}\")\n",
                "\n",
                "# Save model (as requested: xgboost_model.pkl)\n",
                "joblib.dump(best_model, \"../models/xgboost_model.pkl\")\n",
                "print(f\"Saved {best_model_name} model to ../models/xgboost_model.pkl\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Plot Feature Importance (for tree-based models)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if hasattr(best_model, 'feature_importances_'):\n",
                "    importances = best_model.feature_importances_\n",
                "    indices = np.argsort(importances)[::-1]\n",
                "    \n",
                "    plt.figure(figsize=(10, 6))\n",
                "    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')\n",
                "    plt.title(f\"Feature Importances - {best_model_name}\")\n",
                "    plt.xlabel(\"Relative Importance\")\n",
                "    plt.tight_layout()\n",
                "    plt.savefig(\"../images/feature_importance.png\", dpi=150)\n",
                "    plt.show()"
            ]
        }
    ]
    create_notebook("notebooks/03_Model_Training.ipynb", cells)

def main():
    build_eda_notebook()
    build_preprocessing_notebook()
    build_training_notebook()

if __name__ == "__main__":
    main()
