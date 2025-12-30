import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List

# Configure page
st.set_page_config(
    page_title="🏥 Hospital Demand Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        # Load forecasts
        forecasts = {}
        forecast_dir = "models/forecasts"
        if os.path.exists(forecast_dir):
            for file in os.listdir(forecast_dir):
                if file.endswith('.csv'):
                    location = file.replace('forecast_', '').replace('.csv', '').replace('_', ' ').title()
                    forecasts[location] = pd.read_csv(os.path.join(forecast_dir, file))
        
        # Load performance metrics
        performance_metrics = {}
        if os.path.exists("models/performance_metrics.json"):
            with open("models/performance_metrics.json", 'r') as f:
                performance_metrics = json.load(f)
        
        # Load historical data
        historical_data = pd.DataFrame()
        if os.path.exists("data/processed/combined_forecast_data.csv"):
            historical_data = pd.read_csv("data/processed/combined_forecast_data.csv")
            historical_data['ds'] = pd.to_datetime(historical_data['ds'])
        
        # Load capacity data
        capacity_data = pd.DataFrame()
        db_path = "data/hospital_forecasting.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            capacity_data = pd.read_sql_query("SELECT * FROM hospital_capacity", conn)
            conn.close()
        
        return forecasts, performance_metrics, historical_data, capacity_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}, {}, pd.DataFrame(), pd.DataFrame()

def create_forecast_chart(forecast_df: pd.DataFrame, location: str, days_to_show: int = 60):
    """Create forecast visualization"""
    if forecast_df.empty:
        return None
    
    # Convert date column
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    # Get recent data
    end_date = forecast_df['ds'].max()
    start_date = end_date - timedelta(days=days_to_show)
    recent_data = forecast_df[forecast_df['ds'] >= start_date].copy()
    
    # Determine forecast start
    today = datetime.now().date()
    forecast_start_idx = recent_data[recent_data['ds'].dt.date <= today].index.max()
    
    fig = go.Figure()
    
    # Historical data
    if forecast_start_idx is not None:
        historical = recent_data.loc[:forecast_start_idx]
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['yhat'],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=2)
        ))
    
    # Forecast data
    if forecast_start_idx is not None:
        forecast = recent_data.loc[forecast_start_idx:]
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(255,127,14,0.3)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='rgba(255,127,14,0.3)', width=0),
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"🏥 Hospital Admission Forecast - {location}",
        xaxis_title="Date",
        yaxis_title="Daily Admissions",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def calculate_alert_level(predicted_admissions: float, capacity: int) -> str:
    """Calculate alert level based on predicted admissions and capacity"""
    if capacity == 0:
        return "unknown"
    
    utilization = predicted_admissions / capacity
    
    if utilization > 0.85:
        return "high"
    elif utilization > 0.70:
        return "medium"
    else:
        return "low"

def display_alerts(forecasts: Dict, capacity_data: pd.DataFrame):
    """Display capacity alerts"""
    st.subheader("🚨 Capacity Alerts")
    
    if capacity_data.empty:
        st.warning("No capacity data available for alerts")
        return
    
    alerts = []
    
    for location, forecast_df in forecasts.items():
        if forecast_df.empty:
            continue
            
        # Get latest forecast
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        today = datetime.now().date()
        future_forecast = forecast_df[forecast_df['ds'].dt.date > today]
        
        if future_forecast.empty:
            continue
        
        # Get next 7 days average
        next_week = future_forecast.head(7)
        avg_predicted = next_week['yhat'].mean()
        
        # Get capacity for this location
        location_capacity = capacity_data[capacity_data['location'] == location]
        if location_capacity.empty:
            continue
        
        total_beds = location_capacity['total_beds'].iloc[0]
        alert_level = calculate_alert_level(avg_predicted, total_beds)
        
        alerts.append({
            'location': location,
            'predicted_admissions': avg_predicted,
            'total_beds': total_beds,
            'utilization': avg_predicted / total_beds if total_beds > 0 else 0,
            'alert_level': alert_level
        })
    
    # Sort by utilization (highest first)
    alerts.sort(key=lambda x: x['utilization'], reverse=True)
    
    # Display alerts
    for alert in alerts[:5]:  # Show top 5
        utilization_pct = alert['utilization'] * 100
        
        if alert['alert_level'] == 'high':
            st.markdown(f"""
            <div class="alert-high">
                <strong>🔴 HIGH ALERT - {alert['location']}</strong><br>
                Predicted: {alert['predicted_admissions']:.0f} admissions/day<br>
                Capacity: {alert['total_beds']} beds<br>
                Utilization: {utilization_pct:.1f}%
            </div>
            """, unsafe_allow_html=True)
        elif alert['alert_level'] == 'medium':
            st.markdown(f"""
            <div class="alert-medium">
                <strong>🟡 MEDIUM ALERT - {alert['location']}</strong><br>
                Predicted: {alert['predicted_admissions']:.0f} admissions/day<br>
                Capacity: {alert['total_beds']} beds<br>
                Utilization: {utilization_pct:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                <strong>🟢 LOW RISK - {alert['location']}</strong><br>
                Predicted: {alert['predicted_admissions']:.0f} admissions/day<br>
                Capacity: {alert['total_beds']} beds<br>
                Utilization: {utilization_pct:.1f}%
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🏥 Hospital Demand Forecasting Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        forecasts, performance_metrics, historical_data, capacity_data = load_data()
    
    if not forecasts:
        st.error("No forecast data available. Please run the forecasting models first.")
        st.code("python src/models/prophet_forecasting.py")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Location selector
    available_locations = list(forecasts.keys())
    selected_location = st.sidebar.selectbox(
        "Select Location",
        available_locations,
        index=0 if available_locations else None
    )
    
    # Date range selector
    days_to_show = st.sidebar.slider("Days to Show", 30, 180, 60)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    if selected_location and selected_location in forecasts:
        
        # Performance metrics
        if selected_location in performance_metrics:
            metrics = performance_metrics[selected_location]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics['accuracy']:.1f}%",
                    delta=f"MAPE: {metrics['mape']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Avg Daily Admissions",
                    f"{metrics['mean_actual']:.0f}",
                    delta=f"Predicted: {metrics['mean_predicted']:.0f}"
                )
            
            with col3:
                st.metric(
                    "RMSE",
                    f"{metrics['rmse']:.1f}",
                    delta=f"MAE: {metrics['mae']:.1f}"
                )
            
            with col4:
                test_days = metrics.get('test_days', 30)
                st.metric(
                    "Test Period",
                    f"{test_days} days",
                    delta="Validation"
                )
        
        # Forecast chart
        st.subheader(f"📈 Forecast for {selected_location}")
        
        forecast_df = forecasts[selected_location]
        chart = create_forecast_chart(forecast_df, selected_location, days_to_show)
        
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.error("Unable to create forecast chart")
        
        # Forecast table
        st.subheader("📋 Next 14 Days Forecast")
        
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        today = datetime.now().date()
        future_forecast = forecast_df[forecast_df['ds'].dt.date > today].head(14)
        
        if not future_forecast.empty:
            display_df = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            display_df.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Predicted'] = display_df['Predicted'].round(0).astype(int)
            display_df['Lower Bound'] = display_df['Lower Bound'].round(0).astype(int)
            display_df['Upper Bound'] = display_df['Upper Bound'].round(0).astype(int)
            
            st.dataframe(display_df, use_container_width=True)
        
        # Capacity alerts section
        st.divider()
        display_alerts(forecasts, capacity_data)
        
        # Historical trends
        st.divider()
        st.subheader("📊 Historical Trends Analysis")
        
        if not historical_data.empty:
            location_historical = historical_data[historical_data['location'] == selected_location].copy()
            
            if not location_historical.empty:
                # Monthly trend
                location_historical['month'] = pd.to_datetime(location_historical['ds']).dt.to_period('M')
                monthly_avg = location_historical.groupby('month')['y'].mean().reset_index()
                monthly_avg['month'] = monthly_avg['month'].astype(str)
                
                fig_monthly = px.line(
                    monthly_avg, 
                    x='month', 
                    y='y',
                    title=f"Monthly Average Admissions - {selected_location}",
                    labels={'y': 'Average Daily Admissions', 'month': 'Month'}
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Overall summary
    st.divider()
    st.subheader("📊 Overall System Performance")
    
    if performance_metrics:
        # Performance summary
        avg_accuracy = np.mean([m['accuracy'] for m in performance_metrics.values()])
        total_locations = len(performance_metrics)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Locations Covered", total_locations)
        
        with col2:
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        
        with col3:
            high_accuracy = len([m for m in performance_metrics.values() if m['accuracy'] > 80])
            st.metric("High Accuracy Models", f"{high_accuracy}/{total_locations}")
        
        # Performance by location
        perf_df = pd.DataFrame(performance_metrics).T.reset_index()
        perf_df.columns = ['Location'] + list(perf_df.columns[1:])
        
        fig_perf = px.bar(
            perf_df.sort_values('accuracy', ascending=True),
            x='accuracy',
            y='Location',
            orientation='h',
            title="Model Accuracy by Location",
            labels={'accuracy': 'Accuracy (%)'}
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **🏥 Hospital Demand Forecasting System**  
    *Powered by Prophet ML & Streamlit*  
    📧 Contact: your-email@example.com | 📱 GitHub: your-github-profile
    """)

if __name__ == "__main__":
    main()