# Automated Exploratory Data Analysis & AI Reporting Tool

An interactive web application built with Streamlit that automates the Exploratory Data Analysis (EDA) process. Users can upload any CSV dataset to instantly generate statistical summaries, interactive visualizations, and an AI-powered text analysis report.

## Features

* **Instant Data Overview:** Automatically extracts dataset shape, data types, and sample data.
* **Descriptive Statistics:** Calculates numerical summaries and visualizes missing values using gradient tables and bar charts.
* **Interactive Visualizations:** Dynamically generates histograms for numerical features and bar charts for categorical features (handling high-cardinality limits automatically).
* **Correlation Analysis:** Computes and visualizes feature relationships using a masked Pearson correlation heatmap.
* **AI-Driven Insights:** Integrates with the Google Gemini API to generate a comprehensive, downloadable text analysis of the dataset's statistical metadata, highlighting anomalies and key relationships.

## Tech Stack

* **Language:** Python
* **Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express
* **AI Integration:** Google Gemini API (via HTTP requests)

## Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/V0idkun/your-repo-name.git](https://github.com/V0idkun/your-repo-name.git)
   cd your-repo-name
