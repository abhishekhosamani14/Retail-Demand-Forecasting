import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as objects
import streamlit as st
import joblib

# Import prediction helpers
from src.predict import load_model_and_scaler, predict_demand, get_features_for_prediction

# Custom Styling & CSS
st.set_page_config(
    page_title="Retail Demand Forecasting & Price Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d313f;
    }
    .stMetric label {
        color: #8f94a5 !important;
        font-weight: 600;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #4ef0a0 !important;
        font-size: 2.2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("# 📈 Retail Demand Forecasting & Price Optimization Platform")
st.markdown("---")

# Load model, scaler, and historical data
@st.cache_resource
def load_assets():
    model, scaler = load_model_and_scaler()
    return model, scaler

@st.cache_data
def load_historical_data():
    df = pd.read_csv("data/train.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    model, scaler = load_assets()
    historical_df = load_historical_data()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.info("Make sure you have run the training pipeline first: `python -m src.train_model`")
    assets_loaded = False

if assets_loaded:
    # Sidebar inputs
    st.sidebar.header("🎯 Parameters")
    
    # Store ID selection
    stores = sorted(historical_df['store'].unique())
    selected_store = st.sidebar.selectbox("🏬 Select Store ID", stores, index=0)
    
    # Item ID selection
    items = sorted(historical_df['item'].unique())
    selected_item = st.sidebar.selectbox("📦 Select Item ID", items, index=0)
    
    # Date selection (Restrict between min and max date in training, plus support 2026 forecast)
    min_date = historical_df['date'].min().date()
    max_date = pd.to_datetime("2026-12-31").date()
    # Default to the last date in train.csv
    default_date = historical_df['date'].max().date()
    
    selected_date = st.sidebar.date_input(
        "📅 Select Forecast Date", 
        value=default_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Item configuration base price lookup to set realistic slider bounds
    # Item 1: 10, Item 2: 25, Item 3: 5, Item 4: 15, Item 5: 40
    base_prices = {1: 10.0, 2: 25.0, 3: 5.0, 4: 15.0, 5: 40.0}
    item_base_price = base_prices.get(selected_item, 15.0)
    
    # Price input slider
    min_slider = float(round(item_base_price * 0.5, 2))
    max_slider = float(round(item_base_price * 1.8, 2))
    
    # Look up last price in historical data for this item/store as a default
    subset_hist = historical_df[(historical_df['store'] == selected_store) & (historical_df['item'] == selected_item)]
    last_price = float(subset_hist.sort_values('date').iloc[-1]['price']) if not subset_hist.empty else item_base_price
    
    selected_price = st.sidebar.slider(
        "💲 Set Selling Price ($)",
        min_value=min_slider,
        max_value=max_slider,
        value=last_price,
        step=0.1
    )
    
    # Create prediction
    date_str = selected_date.strftime("%Y-%m-%d")
    prediction = predict_demand(selected_store, selected_item, selected_price, date_str, historical_df, model, scaler)
    expected_revenue = prediction * selected_price
    
    # Main dashboard tabs
    tab1, tab2 = st.tabs(["📊 Demand Forecasting", "💡 Price Optimization"])
    
    with tab1:
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Forecasted Demand (Units)", 
                value=f"{int(round(prediction))}",
                help="Predicted sales quantity for the selected store, item, date, and price."
            )
        with col2:
            st.metric(
                label="Expected Revenue ($)", 
                value=f"${expected_revenue:.2f}",
                help="Calculated as: Predicted Demand × Set Selling Price."
            )
        with col3:
            avg_hist_sales = subset_hist['sales'].mean()
            st.metric(
                label="Historical Avg Demand (Daily)", 
                value=f"{int(round(avg_hist_sales))}",
                help="Mean daily sales for this store and item across all historical data."
            )
            
        st.write("---")
        
        # Charts Grid
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📈 Historical Sales Trend Over Time")
            # Group by month/date for selected store/item
            plot_df = subset_hist.sort_values('date').tail(365) # Show last year of data
            fig_trend = px.line(
                plot_df, 
                x='date', 
                y='sales',
                labels={'sales': 'Sales Units', 'date': 'Date'},
                template='plotly_dark',
                color_discrete_sequence=['#4ef0a0']
            )
            fig_trend.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                xaxis_title=None
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("📅 Monthly Seasonal Sales Analysis")
            monthly_avg = subset_hist.groupby(subset_hist['date'].dt.strftime('%b'))['sales'].mean().reindex(
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ).reset_index()
            
            fig_monthly = px.bar(
                monthly_avg, 
                x='date', 
                y='sales',
                labels={'sales': 'Avg Sales Units', 'date': 'Month'},
                template='plotly_dark',
                color_discrete_sequence=['#00c0f0']
            )
            fig_monthly.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                xaxis_title=None
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
        st.write("---")
        
        # Bottom Grid (Feature Importance and Historical Data Table)
        b1, b2 = st.columns(2)
        
        with b1:
            st.subheader("⚡ Model Feature Importance")
            if hasattr(model, 'feature_importances_'):
                features_ordered = [
                    'Store ID', 'Item ID', 'Price',
                    'Year', 'Month', 'Week', 'Day', 'Day of Week', 'Quarter', 'Is Weekend',
                    'Sales Lag 7', 'Sales Lag 14', 'Sales Lag 30',
                    'Roll Mean 7', 'Roll Mean 14', 'Roll Mean 30'
                ]
                fi_df = pd.DataFrame({
                    'Feature': features_ordered,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_fi = px.bar(
                    fi_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    template='plotly_dark',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_fi.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=350,
                    coloraxis_showscale=False,
                    xaxis_title="Relative Importance",
                    yaxis_title=None
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance is not available for this model type.")
                
        with b2:
            st.subheader("📋 Historical Sales Table (Recent)")
            table_df = subset_hist.sort_values('date', ascending=False).head(30)[['date', 'price', 'sales']]
            table_df['date'] = table_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(table_df, use_container_width=True, height=270)
            
            # Prediction Download
            pred_record = pd.DataFrame([{
                'Date': date_str,
                'Store_ID': selected_store,
                'Item_ID': selected_item,
                'Set_Price': selected_price,
                'Forecasted_Demand': int(round(prediction)),
                'Expected_Revenue': expected_revenue
            }])
            
            # CSV conversion
            csv = pred_record.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction as CSV",
                data=csv,
                file_name=f"demand_forecast_store_{selected_store}_item_{selected_item}_{date_str}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with tab2:
        st.subheader("💡 Price Sensitivity & Revenue Maximization")
        st.write("This tool runs simulations using our trained machine learning model across different price levels to find the optimal price point.")
        
        # Simulation over 30 price points in the slider range
        sim_prices = np.linspace(min_slider, max_slider, 30)
        sim_demands = []
        sim_revenues = []
        
        for p in sim_prices:
            d = predict_demand(selected_store, selected_item, p, date_str, historical_df, model, scaler)
            sim_demands.append(d)
            sim_revenues.append(d * p)
            
        sim_df = pd.DataFrame({
            'Price': sim_prices,
            'Demand': sim_demands,
            'Revenue': sim_revenues
        })
        
        # Find optimal price
        opt_idx = sim_df['Revenue'].idxmax()
        opt_price = sim_df.loc[opt_idx, 'Price']
        opt_demand = sim_df.loc[opt_idx, 'Demand']
        opt_revenue = sim_df.loc[opt_idx, 'Revenue']
        
        # Display recommendations
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.markdown(f"""
            <div style="background-color: #1c3d2e; padding: 20px; border-radius: 10px; border: 1px solid #2e7d32;">
                <h4 style="color: #4ef0a0; margin-top: 0px;">💡 Recommended Optimal Price</h4>
                <p style="font-size: 24px; font-weight: bold; margin-bottom: 5px; color: white;">${opt_price:.2f}</p>
                <p style="margin-bottom: 0; color: #b9f6ca;">Maximizes revenue to <b>${opt_revenue:.2f}</b> with forecasted demand of <b>{int(round(opt_demand))} units</b>.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_rec2:
            rev_diff = opt_revenue - expected_revenue
            percent_increase = (rev_diff / expected_revenue * 100) if expected_revenue > 0 else 0
            
            if abs(selected_price - opt_price) < 0.15:
                st.markdown(f"""
                <div style="background-color: #1a1c24; padding: 20px; border-radius: 10px; border: 1px solid #2d313f;">
                    <h4 style="color: #00c0f0; margin-top: 0px;">⚡ Optimization Insight</h4>
                    <p style="font-size: 24px; font-weight: bold; margin-bottom: 5px; color: white;">Price is Optimized!</p>
                    <p style="margin-bottom: 0; color: #8f94a5;">Your current price of <b>${selected_price:.2f}</b> is at or very near the revenue-maximizing price.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                action = "increase" if selected_price < opt_price else "decrease"
                st.markdown(f"""
                <div style="background-color: #3b2b1a; padding: 20px; border-radius: 10px; border: 1px solid #e65100;">
                    <h4 style="color: #ffb74d; margin-top: 0px;">⚡ Optimization Insight</h4>
                    <p style="font-size: 24px; font-weight: bold; margin-bottom: 5px; color: white;">Potential Revenue Uplift: +{percent_increase:.1f}%</p>
                    <p style="margin-bottom: 0; color: #ffe0b2;">Consider <b>{action}ing</b> your price from <b>${selected_price:.2f}</b> to <b>${opt_price:.2f}</b> to capture an extra <b>${rev_diff:.2f}</b> in daily revenue.</p>
                </div>
                """, unsafe_allow_html=True)
                
        st.write("---")
        
        # Simulation charts
        sc1, sc2 = st.columns(2)
        
        with sc1:
            st.subheader("📉 Price vs. Predicted Demand (Demand Curve)")
            fig_elasticity = px.line(
                sim_df, 
                x='Price', 
                y='Demand',
                labels={'Price': 'Price ($)', 'Demand': 'Predicted Demand (Units)'},
                template='plotly_dark',
                color_discrete_sequence=['#ffa726']
            )
            # Add dot for current selected price
            fig_elasticity.add_trace(objects.Scatter(
                x=[selected_price], 
                y=[prediction],
                mode='markers+text',
                name='Current',
                text=['Current Price'],
                textposition='top right',
                marker=dict(color='#ff1744', size=12, line=dict(color='white', width=2))
            ))
            # Add dot for optimal price
            fig_elasticity.add_trace(objects.Scatter(
                x=[opt_price], 
                y=[opt_demand],
                mode='markers+text',
                name='Optimal',
                text=['Optimal Price'],
                textposition='bottom left',
                marker=dict(color='#00e676', size=12, line=dict(color='white', width=2))
            ))
            fig_elasticity.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
                showlegend=False
            )
            st.plotly_chart(fig_elasticity, use_container_width=True)
            
        with sc2:
            st.subheader("💰 Price vs. Expected Revenue")
            fig_revenue = px.line(
                sim_df, 
                x='Price', 
                y='Revenue',
                labels={'Price': 'Price ($)', 'Revenue': 'Expected Revenue ($)'},
                template='plotly_dark',
                color_discrete_sequence=['#29b6f6']
            )
            # Add dot for current selected price
            fig_revenue.add_trace(objects.Scatter(
                x=[selected_price], 
                y=[expected_revenue],
                mode='markers+text',
                name='Current',
                text=['Current Rev'],
                textposition='top right',
                marker=dict(color='#ff1744', size=12, line=dict(color='white', width=2))
            ))
            # Add dot for optimal price
            fig_revenue.add_trace(objects.Scatter(
                x=[opt_price], 
                y=[opt_revenue],
                mode='markers+text',
                name='Optimal',
                text=['Max Rev'],
                textposition='bottom center',
                marker=dict(color='#00e676', size=12, line=dict(color='white', width=2))
            ))
            fig_revenue.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
                showlegend=False
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
