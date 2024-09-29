# Antimicrobial Resistance (AMR) Data Visualization Dashboard

## Project Overview
This project is an interactive data visualization dashboard built using Python, Plotly, and Streamlit. The dashboard aims to analyze and visualize Antimicrobial Resistance (AMR), Antimicrobial Use (AMU), and Antimicrobial Consumption (AMC) trends using publicly available data from the **Global Antimicrobial Resistance and Use Surveillance System (GLASS)** by WHO.

The project is designed to demonstrate the ability to collate, analyze, and disseminate AMR/AMU/AMC data in line with the requirements for a **Data Analyst position** in the AMR National Repository at the Ministry of Health, Ghana.

## Features
- **Interactive Time Series Visualizations**: Visualize AMR rates over time across different regions.
- **Geospatial Analysis**: Map AMR rates across various countries or regions.
- **Filterable Dashboard**: Users can filter the data based on specific pathogens, regions, or time periods.
- **One-Health Approach**: Integrates data from human, animal, and environmental sources.

## Data Source
The data used in this project comes from the publicly available **GLASS dashboard** hosted by the World Health Organization. The dataset contains information on AMR, AMU, and AMC trends for various countries, including Ghana, and is updated regularly.

You can access the dataset [here](https://data.afro.who.int/en_GB/dataset/global-antimicrobial-resistance-and-use-surveillance-system-glass-dashboard).

## Project Structure
```bash
├── app.py                # Main Streamlit app
├── data                  # Folder containing datasets (AMR, AMC, etc.)
├── assets                # Folder for additional resources (images, logos)
├── README.md             # Project documentation
└── requirements.txt      # List of Python dependencies
