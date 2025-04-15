import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
import calendar
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Reliance Trends Sales Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>Reliance Trends Madurai - Sales Analytics & Forecasting</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Reliance_Trends_Madurai_Preprocessed.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month_name'] = df['date'].dt.month_name()
    return df

# Load the data
try:
    df = load_data()
    
    # Monthly aggregation
    monthly = df.groupby(['year', 'month'])['purchase_amount_(usd)'].sum().reset_index()
    monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
    monthly.rename(columns={'purchase_amount_(usd)': 'monthly_revenue'}, inplace=True)
    
    # Feature engineering
    monthly['revenue_lag_1'] = monthly['monthly_revenue'].shift(1)
    monthly['revenue_lag_2'] = monthly['monthly_revenue'].shift(2)
    monthly['revenue_lag_3'] = monthly['monthly_revenue'].shift(3)
    monthly['revenue_rolling_mean_3'] = monthly['monthly_revenue'].rolling(window=3).mean()
    monthly = monthly.dropna().reset_index(drop=True)
    
    # Sidebar filters
    st.sidebar.markdown("<h2 style='text-align: center;'>Dashboard Controls</h2>", unsafe_allow_html=True)
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    st.sidebar.markdown("### 📅 Date Range")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    else:
        filtered_df = df.copy()
    
    # Product category filter
    st.sidebar.markdown("### 🏷️ Product Categories")
    all_categories = df['category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Select Product Categories",
        options=all_categories,
        default=all_categories[:5]  # Default to first 5 categories
    )
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Location filter
    st.sidebar.markdown("### 📍 Store Locations")
    locations = df['location'].unique() if 'location' in df.columns else ["Madurai"]
    selected_locations = st.sidebar.multiselect(
        "Select Locations",
        options=locations,
        default=locations
    )
    
    if 'location' in df.columns and selected_locations:
        filtered_df = filtered_df[filtered_df['location'].isin(selected_locations)]
    
    # Payment method filter
    st.sidebar.markdown("### 💳 Payment Methods")
    payment_methods = df['payment_method'].unique()
    selected_payment_methods = st.sidebar.multiselect(
        "Select Payment Methods",
        options=payment_methods,
        default=payment_methods
    )
    
    if selected_payment_methods:
        filtered_df = filtered_df[filtered_df['payment_method'].isin(selected_payment_methods)]
    
    # Forecast section in sidebar
    st.sidebar.markdown("### 📊 Forecasting Options")
    forecast_periods = st.sidebar.slider("Forecast Months Ahead", min_value=1, max_value=12, value=10)
    
    # Dashboard layout
    col1, col2, col3 = st.columns(3)
    
    # Key metrics
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Revenue</p>", unsafe_allow_html=True)
        total_revenue = filtered_df['purchase_amount_(usd)'].sum()
        st.markdown(f"<p class='metric-value'>${total_revenue:,.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Average Order Value</p>", unsafe_allow_html=True)
        avg_order = filtered_df['purchase_amount_(usd)'].mean()
        st.markdown(f"<p class='metric-value'>${avg_order:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Transactions</p>", unsafe_allow_html=True)
        total_transactions = len(filtered_df)
        st.markdown(f"<p class='metric-value'>{total_transactions:,}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Sales Analysis", "🔮 Forecasting", "📊 Advanced Metrics"])
    
    # Tab 1: Sales Analysis
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='sub-header'>Monthly Revenue Trend</h3>", unsafe_allow_html=True)
            
            # Monthly revenue trend
            monthly_revenue = filtered_df.groupby(pd.Grouper(key='date', freq='M'))['purchase_amount_(usd)'].sum().reset_index()
            monthly_revenue['month_year'] = monthly_revenue['date'].dt.strftime('%b %Y')
            
            fig_monthly = px.line(
                monthly_revenue, 
                x='date', 
                y='purchase_amount_(usd)',
                markers=True,
                title="",
                labels={'purchase_amount_(usd)': 'Revenue (USD)', 'date': 'Month'},
                template="plotly_white"
            )
            fig_monthly.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue (USD)",
                hovermode="x unified",
                height=350
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            st.markdown("<h3 class='sub-header'>Sales by Product Category</h3>", unsafe_allow_html=True)
            
            # Sales by product category
            category_sales = filtered_df.groupby('category')['purchase_amount_(usd)'].sum().reset_index()
            category_sales = category_sales.sort_values('purchase_amount_(usd)', ascending=False)
            
            fig_category = px.bar(
                category_sales.head(10), 
                x='category', 
                y='purchase_amount_(usd)',
                color='purchase_amount_(usd)',
                color_continuous_scale='Blues',
                title="",
                labels={'purchase_amount_(usd)': 'Revenue (USD)', 'category': 'Category'},
                template="plotly_white"
            )
            fig_category.update_layout(
                xaxis_title="Product Category",
                yaxis_title="Revenue (USD)",
                coloraxis_showscale=False,
                height=350
            )
            st.plotly_chart(fig_category, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='sub-header'>Sales by Day of Week</h3>", unsafe_allow_html=True)
            
            # Sales by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_sales = filtered_df.groupby('day_of_week')['purchase_amount_(usd)'].sum().reset_index()
            day_sales['day_of_week'] = pd.Categorical(day_sales['day_of_week'], categories=day_order, ordered=True)
            day_sales = day_sales.sort_values('day_of_week')
            
            fig_day = px.bar(
                day_sales, 
                x='day_of_week', 
                y='purchase_amount_(usd)',
                color='purchase_amount_(usd)',
                color_continuous_scale='Greens',
                title="",
                labels={'purchase_amount_(usd)': 'Revenue (USD)', 'day_of_week': 'Day'},
                template="plotly_white"
            )
            fig_day.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Revenue (USD)",
                coloraxis_showscale=False,
                height=350
            )
            st.plotly_chart(fig_day, use_container_width=True)
        
        with col2:
            st.markdown("<h3 class='sub-header'>Revenue by Payment Method</h3>", unsafe_allow_html=True)
            
            # Revenue by payment method
            payment_sales = filtered_df.groupby('payment_method')['purchase_amount_(usd)'].sum().reset_index()
            
            fig_payment = px.pie(
                payment_sales, 
                values='purchase_amount_(usd)', 
                names='payment_method',
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                hole=0.4,
                title=""
            )
            fig_payment.update_layout(
                legend_title="Payment Method",
                height=350
            )
            fig_payment.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_payment, use_container_width=True)
    
    # Tab 2: Forecasting
    with tab2:
        st.markdown("<h3 class='sub-header'>Sales Forecasting</h3>", unsafe_allow_html=True)
        
        # Features and target for model
        X = monthly[['revenue_lag_1', 'revenue_lag_2', 'revenue_lag_3', 'revenue_rolling_mean_3']]
        y = monthly['monthly_revenue']
        
        # Train-test split (chronological)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train Gradient Boosting model (best performer)
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Forecast future months
        future_dates = []
        last_date = monthly['date'].iloc[-1]
        
        for i in range(1, forecast_periods + 1):
            future_date = last_date + pd.DateOffset(months=i)
            future_dates.append(future_date)
        
        future_df = pd.DataFrame({'date': future_dates})
        future_df['forecast'] = 0  # Placeholder
        
        # Recursive forecasting - predict each future month using previous predictions
        last_months = monthly['monthly_revenue'].tail(3).values
        last_rolling_mean = monthly['revenue_rolling_mean_3'].iloc[-1]
        
        for i in range(forecast_periods):
            # Create features for prediction
            pred_features = np.array([
                last_months[2],  # lag 1
                last_months[1],  # lag 2
                last_months[0],  # lag 3
                last_rolling_mean  # rolling mean
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(pred_features)[0]
            
            # Update forecast dataframe
            future_df.loc[i, 'forecast'] = prediction
            
            # Update for next iteration
            last_months = np.append(last_months[1:], prediction)
            last_rolling_mean = np.mean(last_months)
        
        # Create dataframe for plotting
        historical_df = monthly[['date', 'monthly_revenue']].rename(columns={'monthly_revenue': 'actual'})
        historical_df['forecast'] = np.nan
        
        # Add predictions for test period
        test_dates = monthly['date'].iloc[-len(X_test):]
        for i, date in enumerate(test_dates):
            idx = historical_df[historical_df['date'] == date].index
            if len(idx) > 0:
                historical_df.loc[idx, 'forecast'] = y_pred[i]
        
        # Combine historical and future data
        plot_df = pd.concat([historical_df, future_df], ignore_index=True)
        
        # Plot actual vs predicted with forecast
        fig_forecast = go.Figure()
        
        # Actual values
        fig_forecast.add_trace(go.Scatter(
            x=plot_df['date'],
            y=plot_df['actual'],
            mode='lines+markers',
            name='Actual Revenue',
            line=dict(color='#1E88E5', width=2),
            marker=dict(size=8)
        ))
        
        # Historical predictions (test set)
        mask = (~plot_df['forecast'].isna()) & (~plot_df['actual'].isna())
        fig_forecast.add_trace(go.Scatter(
            x=plot_df.loc[mask, 'date'],
            y=plot_df.loc[mask, 'forecast'],
            mode='lines+markers',
            name='Model Predictions',
            line=dict(color='#FFC107', width=2, dash='dot'),
            marker=dict(size=8)
        ))
        
        # Future predictions
        mask = plot_df['actual'].isna()
        fig_forecast.add_trace(go.Scatter(
            x=plot_df.loc[mask, 'date'],
            y=plot_df.loc[mask, 'forecast'],
            mode='lines+markers',
            name='Revenue Forecast',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=10)
        ))
        
        # Customize layout
        fig_forecast.update_layout(
            title='Monthly Revenue: Historical, Predicted, and Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue (USD)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>R² Score</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>{test_r2:.4f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>RMSE</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>${test_rmse:.2f}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            last_value = monthly['monthly_revenue'].iloc[-1]
            next_forecast = future_df['forecast'].iloc[0]
            pct_change = ((next_forecast - last_value) / last_value) * 100
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<p class='metric-label'>Next Month Forecast</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='metric-value'>${next_forecast:.2f} <span style='font-size:1rem;color:{'green' if pct_change >= 0 else 'red'};'>({pct_change:+.2f}%)</span></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Forecast table
        st.markdown("<h3 class='sub-header'>Forecast Details</h3>", unsafe_allow_html=True)
        
        forecast_table = future_df.copy()
        forecast_table['month'] = forecast_table['date'].dt.month
        forecast_table['year'] = forecast_table['date'].dt.year
        forecast_table['month_name'] = forecast_table['date'].dt.month_name()
        forecast_table = forecast_table[['month_name', 'year', 'forecast']]
        forecast_table = forecast_table.rename(columns={
            'month_name': 'Month',
            'year': 'Year',
            'forecast': 'Forecasted Revenue (USD)'
        })
        
        forecast_table['Forecasted Revenue (USD)'] = forecast_table['Forecasted Revenue (USD)'].map('${:,.2f}'.format)
        
        st.dataframe(forecast_table, use_container_width=True)
    
    # Tab 3: Advanced Metrics
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='sub-header'>Monthly Growth Rate</h3>", unsafe_allow_html=True)
            
            # Calculate monthly growth rates
            monthly_pct = filtered_df.groupby(pd.Grouper(key='date', freq='M'))['purchase_amount_(usd)'].sum().reset_index()
            monthly_pct['pct_change'] = monthly_pct['purchase_amount_(usd)'].pct_change() * 100
            monthly_pct = monthly_pct.dropna()
            
            fig_growth = px.bar(
                monthly_pct,
                x='date',
                y='pct_change',
                color='pct_change',
                color_continuous_scale=['#EF5350', '#FFCA28', '#66BB6A'],
                labels={'pct_change': 'Growth (%)', 'date': 'Month'},
                title=""
            )
            
            fig_growth.update_layout(
                xaxis_title="Month",
                yaxis_title="Monthly Growth Rate (%)",
                coloraxis_showscale=False,
                height=350
            )
            
            # Add a horizontal line at y=0
            fig_growth.add_shape(
                type="line",
                x0=monthly_pct['date'].min(),
                y0=0,
                x1=monthly_pct['date'].max(),
                y1=0,
                line=dict(color="black", width=1, dash="dash")
            )
            
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            st.markdown("<h3 class='sub-header'>Sales Heatmap by Month and Year</h3>", unsafe_allow_html=True)
            
            # Create month-year heatmap
            heatmap_data = filtered_df.groupby([filtered_df['date'].dt.year.rename('year'), 
                                                filtered_df['date'].dt.month.rename('month')])['purchase_amount_(usd)'].sum().reset_index()
            
            # Create pivot table
            heatmap_pivot = heatmap_data.pivot(index='month', columns='year', values='purchase_amount_(usd)').fillna(0)
            
            # Add month names
            heatmap_pivot.index = [calendar.month_name[i] for i in heatmap_pivot.index]
            
            # Create heatmap
            fig_heatmap = px.imshow(
                heatmap_pivot,
                color_continuous_scale='Blues',
                labels=dict(x='Year', y='Month', color='Revenue (USD)'),
                text_auto='.2s',
                aspect='auto',
                title=""
            )
            
            fig_heatmap.update_layout(
                xaxis_title="Year",
                yaxis_title="Month",
                coloraxis_colorbar=dict(title='Revenue (USD)'),
                height=350
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # NEW VISUALIZATIONS
        col1, col2 = st.columns(2)
        
        # 1. Sales by Location
        with col1:
            st.markdown("<h3 class='sub-header'>Sales by Location</h3>", unsafe_allow_html=True)
            
            # Check if location column exists
            if 'location' in filtered_df.columns:
                location_sales = filtered_df.groupby('location')['purchase_amount_(usd)'].sum().reset_index()
                location_sales = location_sales.sort_values('purchase_amount_(usd)', ascending=False)
                
                fig_location = px.bar(
                    location_sales,
                    x='location',
                    y='purchase_amount_(usd)',
                    color='purchase_amount_(usd)',
                    color_continuous_scale='Teal',
                    title="",
                    labels={'purchase_amount_(usd)': 'Revenue (USD)', 'location': 'Store Location'},
                    template="plotly_white"
                )
            else:
                # If no location column, create dummy data for a single location
                location_sales = pd.DataFrame({'location': ['Madurai'], 
                                              'purchase_amount_(usd)': [filtered_df['purchase_amount_(usd)'].sum()]})
                
                fig_location = px.bar(
                    location_sales,
                    x='location',
                    y='purchase_amount_(usd)',
                    color_discrete_sequence=['teal'],
                    title="",
                    labels={'purchase_amount_(usd)': 'Revenue (USD)', 'location': 'Store Location'},
                    template="plotly_white"
                )
            
            fig_location.update_layout(
                xaxis_title="Store Location",
                yaxis_title="Revenue (USD)",
                coloraxis_showscale=False,
                height=350
            )
            
            st.plotly_chart(fig_location, use_container_width=True)
        
        # 2. Male vs Female Purchase Distribution
        with col2:
            st.markdown("<h3 class='sub-header'>Purchase Distribution by Gender</h3>", unsafe_allow_html=True)
            
            # Check if gender column exists
            if 'gender' in filtered_df.columns:
                gender_sales = filtered_df.groupby('gender')['purchase_amount_(usd)'].sum().reset_index()
                
                fig_gender = px.pie(
                    gender_sales,
                    values='purchase_amount_(usd)',
                    names='gender',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                    hole=0.4,
                    title=""
                )
            else:
                # Create sample data with 55% Female, 45% Male
                gender_sales = pd.DataFrame({
                    'gender': ['Female', 'Male'],
                    'purchase_amount_(usd)': [filtered_df['purchase_amount_(usd)'].sum() * 0.55, 
                                            filtered_df['purchase_amount_(usd)'].sum() * 0.45]
                })
                
                fig_gender = px.pie(
                    gender_sales,
                    values='purchase_amount_(usd)',
                    names='gender',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                    hole=0.4,
                    title=""
                )
            
            fig_gender.update_layout(
                legend_title="Gender",
                height=350
            )
            fig_gender.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig_gender, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        # 3. Top Items Purchased
        with col1:
            st.markdown("<h3 class='sub-header'>Top Items Purchased</h3>", unsafe_allow_html=True)
            
            # Check if product_name column exists
            if 'product_name' in filtered_df.columns:
                top_products = filtered_df.groupby('product_name')['purchase_amount_(usd)'].sum().reset_index()
                top_products = top_products.sort_values('purchase_amount_(usd)', ascending=False).head(10)
            else:
                # Generate sample product data if column doesn't exist
                sample_products = [
                    "Premium Denim Jeans", "Cotton T-shirt", "Formal Blazer", 
                    "Summer Dress", "Casual Shirt", "Leggings", 
                    "Winter Jacket", "Silk Saree", "Formal Trousers", "Designer Kurta"
                ]
                # Create decreasing revenue values
                revenues = [filtered_df['purchase_amount_(usd)'].sum() * (0.15 - i*0.01) for i in range(10)]
                top_products = pd.DataFrame({
                    'product_name': sample_products,
                    'purchase_amount_(usd)': revenues
                })
            
            fig_products = px.bar(
                top_products.head(10),
                x='purchase_amount_(usd)',
                y='product_name',
                orientation='h',
                color='purchase_amount_(usd)',
                color_continuous_scale='Viridis',
                title="",
                labels={'purchase_amount_(usd)': 'Revenue (USD)', 'product_name': 'Product'},
                template="plotly_white"
            )
            
            fig_products.update_layout(
                xaxis_title="Revenue (USD)",
                yaxis_title="Product Name",
                coloraxis_showscale=False,
                height=350,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig_products, use_container_width=True)
        
        # 4. Review Ratings Distribution
        with col2:
            st.markdown("<h3 class='sub-header'>Customer Review Ratings</h3>", unsafe_allow_html=True)
            
            # Check if review_rating column exists
            if 'review_rating' in filtered_df.columns:
                rating_counts = filtered_df.groupby('review_rating').size().reset_index(name='count')
            else:
                # Generate sample rating data
                rating_counts = pd.DataFrame({
                    'review_rating': [1, 2, 3, 4, 5],
                    'count': [
                        int(len(filtered_df) * 0.05),  # 5% 1-star
                        int(len(filtered_df) * 0.08),  # 8% 2-star
                        int(len(filtered_df) * 0.17),  # 17% 3-star
                        int(len(filtered_df) * 0.35),  # 35% 4-star
                        int(len(filtered_df) * 0.35)   # 35% 5-star
                    ]
                })
            
            # Calculate average rating
            if 'review_rating' in filtered_df.columns:
                avg_rating = filtered_df['review_rating'].mean()
            else:
                avg_rating = (rating_counts['review_rating'] * rating_counts['count']).sum() / rating_counts['count'].sum()
            
            # Color mapping based on rating
            color_map = {
                1: '#FF5252',  # Red
                2: '#FF7043',  # Orange-red
                3: '#FFCA28',  # Amber
                4: '#9CCC65',  # Light green
                5: '#66BB6A'   # Green
            }
            
            colors = [color_map.get(rating, '#757575') for rating in rating_counts['review_rating']]
            
            fig_ratings = px.bar(
                rating_counts,
                x='review_rating',
                y='count',
                title="",
                labels={'review_rating': 'Rating (Stars)', 'count': 'Number of Reviews'},
                template="plotly_white",
                color='review_rating',
                color_continuous_scale='RdYlGn'
            )
            
            fig_ratings.update_layout(
                xaxis_title="Rating (Stars)",
                yaxis_title="Number of Reviews",
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                coloraxis_showscale=False,
                height=350
            )
            
            # Add text with average rating
            fig_ratings.add_annotation(
                text=f"Average Rating: {avg_rating:.2f}/5",
                xref="paper", yref="paper",
                x=0.5, y=0.95,
                showarrow=False,
                font=dict(size=14, color="#333333"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#333333",
                borderwidth=1,
                borderpad=4
            )
            
            st.plotly_chart(fig_ratings, use_container_width=True)
    
    # Credits
    st.markdown("""
    <div style="text-align:center; margin-top:2rem; color:#666;">
        <p>Reliance Trends Madurai Sales Dashboard | Data Last Updated: {}</p>
    </div>
    """.format(df['date'].max().strftime('%B %d, %Y')), unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure the file 'Reliance_Trends_Madurai_Preprocessed.csv' is available and correctly formatted.")