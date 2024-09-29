import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import base64


# Set up page configuration with custom theme
st.set_page_config(layout="wide", page_title="Advanced Antimicrobial Use Dashboard", page_icon="🌍")

st.markdown("""
    <style>
    div[role="tablist"] > div[role="tab"] {
        font-weight: bold;
        font-size: 25px;
    }
    </style>
    """, unsafe_allow_html=True)


# Caching data for optimized performance
@st.cache_data(ttl=600)
def load_data():
    with st.spinner("Loading data..."):
        df_atc4 = pd.read_excel('GLASS-AMU_2016-22_dataset.xlsx', sheet_name='Antimicrobial_Use_ATC4')
        df_aware = pd.read_excel('GLASS-AMU_2016-22_dataset.xlsx', sheet_name='Antibiotic_Use_AWaRe')
    return df_atc4, df_aware

df_atc4, df_aware = load_data()

# Sidebar filter functions with 'Apply Filters' and 'Select All' options
def render_filters(df):
    st.sidebar.title("🔍 Filter Data")
    
    # Step 1: Year Filter
    available_years = sorted(df['Year'].unique())
    select_all_years = st.sidebar.checkbox('Select All Years')
    year_filter = st.sidebar.multiselect('Step 1: Select Year', available_years, default=available_years if select_all_years else None, help="Select the year(s) of interest")
    
    # Step 2: Country Filter
    if year_filter:
        available_countries = df[df['Year'].isin(year_filter)]['CountryTerritoryArea'].unique()
        select_all_countries = st.sidebar.checkbox('Select All Countries')
        country_filter = st.sidebar.multiselect('Step 2: Select Country', available_countries, default=available_countries if select_all_countries else None, help="Select countries")
    else:
        country_filter = []

    # Step 3: Region Filter
    if country_filter:
        available_regions = df[(df['Year'].isin(year_filter)) & (df['CountryTerritoryArea'].isin(country_filter))]['WHORegionName'].unique()
        select_all_regions = st.sidebar.checkbox('Select All Regions')
        region_filter = st.sidebar.multiselect('Step 3: Select WHO Region', available_regions, default=available_regions if select_all_regions else None, help="Select regions")
    else:
        region_filter = []

    # Step 4: Pathogen Filter
    if region_filter:
        available_pathogens = df[(df['Year'].isin(year_filter)) & (df['CountryTerritoryArea'].isin(country_filter)) & 
                                  (df['WHORegionName'].isin(region_filter))]['ATC4Name'].unique()
        select_all_pathogens = st.sidebar.checkbox('Select All Pathogens/Antimicrobial Classes')
        pathogen_filter = st.sidebar.multiselect('Step 4: Select Pathogen/Antimicrobial Class', available_pathogens, default=available_pathogens if select_all_pathogens else None, help="Select pathogen or antimicrobial class")
    else:
        pathogen_filter = []

    # Step 5: Route of Administration Filter
    if pathogen_filter:
        available_routes = df[(df['Year'].isin(year_filter)) & (df['CountryTerritoryArea'].isin(country_filter)) & 
                              (df['WHORegionName'].isin(region_filter)) & (df['ATC4Name'].isin(pathogen_filter))]['RouteOfAdministration'].unique()
        select_all_routes = st.sidebar.checkbox('Select All Routes of Administration')
        route_filter = st.sidebar.multiselect('Step 5: Select Route of Administration', available_routes, default=available_routes if select_all_routes else None, help="Select route of administration")
    else:
        route_filter = []

    # Step 6: Income Level Filter
    if route_filter:
        available_incomes = df[(df['Year'].isin(year_filter)) & (df['CountryTerritoryArea'].isin(country_filter)) & 
                               (df['WHORegionName'].isin(region_filter)) & (df['ATC4Name'].isin(pathogen_filter)) & 
                               (df['RouteOfAdministration'].isin(route_filter))]['IncomeWorldBankJune'].unique()
        select_all_incomes = st.sidebar.checkbox('Select All Income Levels')
        income_filter = st.sidebar.multiselect('Step 6: Select Income Level', available_incomes, default=available_incomes if select_all_incomes else None, help="Select income level")
    else:
        income_filter = []

    # Add an 'Apply Filters' button to process the selection interactively
    if st.sidebar.button("Apply Filters"):
        # Filter dataset based on selections
        df_filtered = df[
            (df['Year'].isin(year_filter)) & 
            (df['CountryTerritoryArea'].isin(country_filter)) & 
            (df['WHORegionName'].isin(region_filter)) & 
            (df['ATC4Name'].isin(pathogen_filter)) &
            (df['RouteOfAdministration'].isin(route_filter)) &
            (df['IncomeWorldBankJune'].isin(income_filter))
        ] if year_filter else pd.DataFrame()  # Return empty DataFrame if no year is selected
    else:
        df_filtered = pd.DataFrame()  # Empty until filters are applied
    
    return df_filtered

# Project Overview and Dataset Information
def display_project_overview():
    st.title("Antimicrobial Resistance (AMR) Data Visualization Dashboard")
    
    st.markdown("""
    ## Project Overview
    This project is an interactive data visualization dashboard built using Python, Plotly, and Streamlit. 
    It aims to analyze and visualize Antimicrobial Resistance (AMR), Antimicrobial Use (AMU), and Antimicrobial Consumption (AMC) trends 
    using publicly available data from the **Global Antimicrobial Resistance and Use Surveillance System (GLASS)** by WHO.
    
    The project demonstrates the ability to collate, analyze, and disseminate AMR/AMU/AMC data, which is critical for the 
    **Data Analyst position** in the AMR National Repository.

    ### About the Author:
    **Michael Adu,PharmD** is a Pharmacist and Data Scientist with expertise in public health data analysis, specializing in antimicrobial resistance (AMR) and 
    antimicrobial use (AMU) data. This dashboard reflects my proficiency in using data visualization tools like Python and Streamlit, 
    as well as my understanding of global health data trends.

    **Relevant Skills:**
    - Data analysis and visualization using Python, Pandas, Plotly, and Streamlit.
    - Proficient in AMR/AMU surveillance, geospatial analysis, and forecasting with Prophet.
    - Experience with public health datasets from WHO and other global organizations.

    **Contact Information:**
    - LinkedIn: https://www.linkedin.com/in/drmichael-adu/
                
    ### Key Features:
    - **Interactive Time Series Visualizations**: Visualize AMU and resistance trends over time across different regions.
    - **Geospatial Analysis**: Map AMU rates across various countries or regions.
    - **Filterable Dashboard**: Users can filter the data based on specific pathogens, regions, or time periods.
    - **One-Health Approach**: Integrates data from human, animal, and environmental sources for holistic analysis.
    
    ### Data Source:
    The data used in this project comes from the publicly available **GLASS dashboard** hosted by the World Health Organization. 
    The dataset contains information on AMR, AMU, and AMC trends for various countries, including Ghana, and is updated regularly.
    
    You can access the dataset [here](https://data.afro.who.int/en_GB/dataset/global-antimicrobial-resistance-and-use-surveillance-system-glass-dashboard).
    """)

# Visualization 1: Comparative Bar Charts for Antimicrobial Classes
def comparative_bar_charts(df_filtered):
    st.markdown("""
    ### Comparative Bar Charts for Antimicrobial Classes
    This visualization shows the comparison of antimicrobial usage across various countries and antimicrobial classes. 
    It helps identify the usage patterns of different antimicrobial agents, providing insights into how antimicrobials are consumed in different regions.
    """)
    if not df_filtered.empty:
        df_grouped = df_filtered.groupby(['CountryTerritoryArea', 'ATC4Name'])['DID'].mean().reset_index()
        fig = px.bar(df_grouped, x='CountryTerritoryArea', y='DID', color='ATC4Name', title="Antimicrobial Use by Class")
        st.plotly_chart(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 2: Geospatial Heatmaps
def geospatial_heatmaps(df_filtered):
    st.markdown("""
    ### Geospatial Heatmaps for Antimicrobial Use
    This visualization presents the geographical distribution of antimicrobial use over time. 
    By looking at different countries or regions, users can explore the spatial patterns of antimicrobial consumption.
    """)
    if not df_filtered.empty:
        geo_data = df_filtered.groupby(['CountryTerritoryArea', 'Year'])[['DID']].mean().reset_index()
        fig = px.choropleth(geo_data, 
                            locations="CountryTerritoryArea", 
                            locationmode='country names',
                            color="DID",
                            hover_name="CountryTerritoryArea",
                            animation_frame="Year",
                            color_continuous_scale="Viridis",
                            title="Global Antimicrobial Use (DID) Over Time",
                            labels={"DID": "Doses per 1,000 inhabitants per day", "Year": "Year"})
        st.plotly_chart(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 3: Time Series Analysis of Antimicrobial Use by Class
def time_series_analysis(df_filtered):
    st.markdown("""
    ### Time Series Analysis of Antimicrobial Use
    This visualization helps explore antimicrobial consumption trends     over time, broken down by different classes of antimicrobials.
    It provides insight into how antimicrobial usage changes across the years and can help track whether antimicrobial use is increasing or decreasing in specific regions or globally.
    """)
    if not df_filtered.empty:
        pathogen_trends = df_filtered.groupby(['Year', 'ATC4Name'])[['DID']].mean().reset_index()
        fig = px.line(pathogen_trends, x='Year', y='DID', color='ATC4Name', title="Trends of Antimicrobial Use by Pathogen Over Time")
        st.plotly_chart(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 4: Boxplots for Antimicrobial Use by Income Level
def boxplots_income_level(df_filtered):
    st.markdown("""
    ### Boxplots for Antimicrobial Use by Income Level
    This visualization compares antimicrobial use across countries grouped by their World Bank income classifications (low, middle, high-income).
    It offers insights into whether income levels impact the amount of antimicrobial use in different countries.
    """)
    if not df_filtered.empty:
        fig = px.box(df_filtered, x='IncomeWorldBankJune', y='DID', title="Antimicrobial Use by Income Level")
        st.plotly_chart(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 5: AWaRe Category Breakdown
def aware_category_breakdown(df_aware):
    st.markdown("""
    ### AWaRe Category Breakdown
    The **AWaRe** (Access, Watch, and Reserve) categorization of antibiotics helps track global antibiotic consumption 
    across these three critical categories. This pie chart breaks down the use of antibiotics into these categories, 
    offering insights into which antibiotics are most used and how global health policies are shaping consumption.
    """)
    aware_summary = df_aware.groupby('AWaReLabel')[['DDD']].sum().reset_index()
    fig = px.pie(aware_summary, values='DDD', names='AWaReLabel', title="Antimicrobial Consumption by AWaRe Category")
    st.plotly_chart(fig)

# Visualization 6: Correlation Heatmaps
def correlation_heatmap(df_filtered):
    st.markdown("""
    ### Correlation Heatmap
    This heatmap shows the correlation between different variables in the dataset, such as Defined Daily Doses (DDD) and 
    Doses per 1,000 inhabitants per day (DID). It helps reveal relationships between antimicrobial usage metrics and 
    can highlight patterns that may warrant further investigation.
    """)
    if not df_filtered.empty:
        correlation_matrix = df_filtered[['DID', 'DDD']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 7: Route of Administration Comparison
def route_of_administration_comparison(df_filtered):
    st.markdown("""
    ### Route of Administration Comparison
    This violin plot shows the distribution of antimicrobial use by the route of administration (e.g., oral, intravenous, etc.).
    It highlights which routes are most common for delivering antimicrobial agents and how their usage varies between countries and regions.
    """)
    if not df_filtered.empty:
        fig = px.violin(df_filtered, y='DID', x='RouteOfAdministration', box=True, points="all", title="DID Distribution by Route of Administration")
        st.plotly_chart(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 8: Outlier Detection for Antimicrobial Use
def outlier_detection(df_filtered):
    st.markdown("""
    ### Outlier Detection in Antimicrobial Use
    Using machine learning (Isolation Forest), this visualization identifies potential outliers in antimicrobial use.
    Outliers may indicate countries or regions where antimicrobial use is significantly higher or lower than expected, 
    which can help target interventions or further investigation.
    """)
    if not df_filtered.empty:
        iso_forest = IsolationForest(contamination=0.05)
        df_filtered['outlier'] = iso_forest.fit_predict(df_filtered[['DID']])
        fig_outliers = px.scatter(df_filtered, x='Year', y='DID', color='outlier', title="Outlier Detection in Antimicrobial Use")
        st.plotly_chart(fig_outliers)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 9: Income Level vs. Antimicrobial Use Trend Analysis
def income_vs_antimicrobial_use(df_filtered):
    st.markdown("""
    ### Income Level vs. Antimicrobial Use Trend
    This line chart visualizes trends in antimicrobial use over time, broken down by income levels (low, middle, high-income).
    It helps identify how economic status may influence antimicrobial use trends and whether specific income groups 
    exhibit different usage patterns over time.
    """)
    if not df_filtered.empty:
        df_grouped = df_filtered.groupby(['Year', 'IncomeWorldBankJune'])[['DID']].mean().reset_index()
        fig = px.line(df_grouped, x='Year', y='DID', color='IncomeWorldBankJune', title="Income Level vs. Antimicrobial Use Over Time")
        st.plotly_chart(fig)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 10: Antimicrobial Stewardship and Forecasting
def stewardship_and_forecasting(df_filtered):
    st.markdown("""
    ### Antimicrobial Stewardship and Forecasting
    This section uses the Prophet forecasting model to predict future antimicrobial use based on historical data.
    Forecasting helps policymakers and healthcare professionals anticipate future trends in antimicrobial consumption, 
    allowing for proactive interventions and better resource allocation.
    """)
    if not df_filtered.empty:
        forecast_data = df_filtered[['Year', 'DID']].groupby('Year').mean().reset_index()
        forecast_data.rename(columns={'Year': 'ds', 'DID': 'y'}, inplace=True)
        
        # Forecasting using Prophet
        periods = st.sidebar.slider("Forecast Periods (Years)", 1, 10, 5)
        prophet_model = Prophet()
        prophet_model.fit(forecast_data)
        
        future = prophet_model.make_future_dataframe(periods=periods, freq='Y')
        forecast = prophet_model.predict(future)
        
        fig_forecast = prophet_model.plot(forecast)
        st.pyplot(fig_forecast)
    else:
        st.warning("No data available. Please adjust your filters.")

# Visualization 11: Custom Reports Based on Filters
def custom_reports(df_filtered):
    st.markdown("""
    ### Custom Reports
    This section allows you to download filtered data based on the parameters you've selected. 
    You can use the filtered data for further analysis or reporting purposes.
    """)
    if not df_filtered.empty:
        st.markdown("### Download Filtered Data")
        csv = df_filtered.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No data available. Please adjust your filters.")

# Original Time Series Decomposition (from previous code)
def time_series_decomposition(df_filtered):
    st.markdown("""
    ### Time Series Decomposition
    This analysis decomposes the time series data for antimicrobial use into its component parts: trend, seasonality, and residuals.
    It helps identify underlying patterns and anomalies that might otherwise be hidden in the raw time series data.
    """)
    if not df_filtered.empty:
        time_series_data = df_filtered.groupby('Year')['DID'].mean().reset_index()
        
        if len(time_series_data) >= 4:
            decomposition = seasonal_decompose(time_series_data['DID'], period=1, model='additive')

            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            axes[0].plot(decomposition.observed)
            axes[0].set_title('Observed')
            axes[1].plot(decomposition.trend)
            axes[1].set_title('Trend')
            axes[2].plot(decomposition.seasonal)
            axes[2].set_title('Seasonal')
            axes[3].plot(decomposition.resid)
            axes[3].set_title('Residuals')

            st.pyplot(fig)
        else:
            st.warning("Not enough data points for time series decomposition.")
    else:
        st.warning("No data available. Please adjust your filters.")

# Original Summary Statistics and Benchmarks (from previous code)
def summary_statistics(df_filtered):
    st.markdown("""
    ### Summary Statistics and Benchmarks
    This section provides a quick summary of the data, such as mean, standard deviation, and other relevant statistics.
    Benchmarks are also displayed to help compare average antimicrobial use with pre-set thresholds, allowing for easy identification of discrepancies.
    """)
    if not df_filtered.empty:
        summary_stats = df_filtered[['DID', 'DDD']].describe()
        st.write(summary_stats)

        benchmark_did = 10  # Arbitrary benchmark for DID
        avg_did = df_filtered['DID'].mean()
        st.metric(label="Average DID", value=f"{avg_did:.2f}", delta=f"{avg_did-benchmark_did:.2f} from benchmark")

        benchmark_ddd = 100  # Arbitrary benchmark for DDD
        avg_ddd = df_filtered['DDD'].mean()
        st.metric(label="Average DDD", value=f"{avg_ddd:.2f}", delta=f"{avg_ddd-benchmark_ddd:.2f} from benchmark")
    else:
        st.warning("No data available. Please adjust your filters.")

# Tabs Rendering Logic using combined tabs
def render_tabs(df_filtered, df_aware):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Geospatial Analysis", "Trends and Analysis", 
        "Advanced Analytics", "Forecasting and Reports"
    ])
    
    with tab1:
        st.subheader("Overview")
        comparative_bar_charts(df_filtered)
        summary_statistics(df_filtered)
    
    with tab2:
        st.subheader("Geospatial Analysis")
        geospatial_heatmaps(df_filtered)
        outlier_detection(df_filtered)
    
    with tab3:
        st.subheader("Trends and Analysis")
        time_series_analysis(df_filtered)
        boxplots_income_level(df_filtered)
        income_vs_antimicrobial_use(df_filtered)
    
    with tab4:
        st.subheader("Advanced Analytics")
        correlation_heatmap(df_filtered)
        route_of_administration_comparison(df_filtered)
        aware_category_breakdown(df_aware)
    
    with tab5:
        st.subheader("Forecasting and Reports")
        stewardship_and_forecasting(df_filtered)
        custom_reports(df_filtered)
        time_series_decomposition(df_filtered)

# Main Function
def main():
    # Display project overview at the top of the dashboard
    display_project_overview()
    
    # Display filters and render the visualizations based on user selection
    df_filtered = render_filters(df_atc4)
    
    if df_filtered.empty:
        st.warning("Please select filters and click 'Apply Filters' to display the data.")
    else:
        render_tabs(df_filtered, df_aware)

# Run the main function
if __name__ == "__main__":
    main()

