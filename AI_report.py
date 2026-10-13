import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests # Needed for API calls
import json # Needed for API payload and response parsing
import time # Needed for exponential backoff

# Constants
# Use the specified model for text generation and Google Search grounding
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
API_KEY = st.secrets["GEMINI_API_KEY"] 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"


# Utility Functions

# Function to safely load data
@st.cache_data
def load_data(uploaded_file):
    """Loads CSV data into a pandas DataFrame."""
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file. Please ensure it is a valid CSV format: {e}")
        return None

# Function to call Gemini API with exponential backoff
def call_gemini_api(system_prompt, user_query, max_retries=5):
    """Handles the Gemini API call with exponential backoff."""
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "tools": [{"google_search": {}}], # Enable search grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                return candidate['content']['parts'][0]['text']
            
            # If successful response but no text, retry or break
            st.error("API call succeeded but generated no text. Retrying...")
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # st.warning(f"API request failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"API request failed after {max_retries} attempts: {e}")
                return "Error: Could not generate AI report due to API connection failure or service error."

    return "Error: API call failed to return content after multiple attempts."


def get_data_metadata_string(df):
    """Prepares key metadata summaries as a single string for the prompt."""
    
    metadata_parts = []
    
    # Dataset Shape
    metadata_parts.append(f"## Dataset Shape:\n- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}")

    # Data Types
    dtype_summary = df.dtypes.astype(str).reset_index()
    dtype_summary.columns = ['Feature', 'Data Type']
    metadata_parts.append("\n## Feature Data Types:\n" + dtype_summary.to_markdown(index=False))

    # 3. Missing Value Analysis
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Feature', 'Missing Count']
    missing_data['Missing Percent'] = (missing_data['Missing Count'] / len(df)) * 100
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
        by='Missing Percent', ascending=False
    )
    if not missing_data.empty:
        metadata_parts.append("\n## Missing Value Summary:\n" + missing_data.to_markdown(index=False))
    else:
        metadata_parts.append("\n## Missing Value Summary:\n- No missing values found.")

    # Numerical Descriptive Stats
    numerical_df = df.select_dtypes(include=np.number)
    if not numerical_df.empty:
        stats_df = numerical_df.describe().T.round(2)
        metadata_parts.append("\n## Numerical Feature Statistics:\n" + stats_df.to_markdown())
    else:
        metadata_parts.append("\n## Numerical Feature Statistics:\n- No numerical features found.")

    # Correlation Analysis
    if not numerical_df.empty and numerical_df.shape[1] > 1:
        corr_matrix = numerical_df.corr().round(2)
        metadata_parts.append("\n## Feature Correlation Matrix (Pearson):\n" + corr_matrix.to_markdown())
    else:
        metadata_parts.append("\n## Feature Correlation Matrix:\n- Not applicable (less than two numerical features).")
        
    return "\n\n".join(metadata_parts)


def generate_ai_report(df):
    """
    Generates a full AI-driven analysis report by summarizing dataset metadata.
    Returns the report text.
    """
    metadata_string = get_data_metadata_string(df)

    system_prompt = (
        "You are a Senior Data Scientist specializing in exploratory data analysis (EDA). "
        "Your task is to analyze the provided dataset statistics and generate a comprehensive, "
        "well-structured, and insightful report. The report must be written in professional, "
        "easy-to-read prose, using Markdown formatting (headings, bolding) but without code blocks. "
        "Focus on: 1) Data Integrity (missing values), 2) Key Variable Insights (distributions, means, anomalies), "
        "3) Feature Relationships (high correlations). Do not make up variable meanings; only use the statistical facts."
    )
    
    user_query = (
        "Please generate a detailed data analysis report based ONLY on the following statistical summaries. "
        "The report should be a self-contained document suitable for immediate sharing. "
        "Do not output anything before the report title. "
        "The report should be titled: 'Automated Exploratory Data Analysis Report'.\n\n"
        "--- DATA SUMMARIES ---\n\n"
        f"{metadata_string}"
    )

    with st.spinner("🧠 Generating AI Analysis Report... This may take a moment."):
        report_text = call_gemini_api(system_prompt, user_query)
        
    return report_text


# Function to generate the comprehensive analysis report (modified to include Section 5)
def generate_report(df, categorical_threshold=20):
    """Generates and displays the full data analysis report."""

    # Pre-calculate data needed for Section 5 and reuse it in other sections
    numerical_df = df.select_dtypes(include=np.number)
    corr_matrix = None
    if not numerical_df.empty and numerical_df.shape[1] > 1:
        corr_matrix = numerical_df.corr().round(2)


    st.header("1. Data Overview")
    st.markdown("A quick look at the dataset's structure, columns, and sample data.")

    # 1.1 Dataset Shape and Size
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    
    # 1.2 Column Information
    with st.expander("Column Names and Data Types"):
        info_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
        })
        st.dataframe(info_df, use_container_width=True)

    # 1.3 Sample Data
    with st.expander("First 5 Rows (Head)"):
        st.dataframe(df.head(), use_container_width=True)


    st.header("2. Descriptive Statistics")
    st.markdown("Statistical summary and missing value analysis.")

    # 2.1 Numerical Descriptive Stats
    with st.expander("Numerical Feature Statistics"):
        if not numerical_df.empty:
            st.dataframe(numerical_df.describe().T, use_container_width=True)
        else:
            st.info("No numerical features found for summary statistics.")
            
    # Missing Value Analysis
    with st.expander("Missing Value Analysis"):
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Feature', 'Missing Count']
        missing_data['Missing Percent'] = (missing_data['Missing Count'] / len(df)) * 100
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
            by='Missing Percent', ascending=False
        )

        if not missing_data.empty:
            st.markdown(f"**Total features with missing values:** `{len(missing_data)}`")
            st.dataframe(missing_data.style.background_gradient(subset=['Missing Percent'], cmap='Reds'), use_container_width=True)

            fig = px.bar(
                missing_data,
                x='Feature',
                y='Missing Percent',
                title='Percentage of Missing Values per Feature',
                color='Missing Percent',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")

    st.header("3. Feature Distributions & Cardinality")
    st.markdown("Visualizations for each feature to understand its distribution.")

    # Feature Iteration
    for col in df.columns:
        unique_count = df[col].nunique()
        is_categorical = df[col].dtype == 'object' or unique_count < categorical_threshold

        with st.container(border=True):
            st.subheader(f"Feature: `{col}`")
            st.write(f"**Cardinality:** `{unique_count}` unique values.")

            if is_categorical:
                # Categorical Features: Bar Chart (Value Counts)
                st.markdown("##### Categorical Feature (Bar Chart)")
                if unique_count <= 50: # Limit plotting very high cardinality text features
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'Count']
                    fig = px.bar(
                        counts, 
                        x=col, 
                        y='Count', 
                        title=f'Count of {col} Categories',
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Skipping bar chart due to high cardinality (>50 unique values).")
            
            elif df[col].dtype in ['int64', 'float64']:
                # Numerical Features: Histogram
                st.markdown("##### Numerical Feature (Histogram)")
                fig = px.histogram(
                    df, 
                    x=col, 
                    marginal="box", # Adds a box plot marginal for distribution view
                    title=f'Distribution of {col}',
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)


    st.header("4. Correlation Analysis")
    st.markdown("Heatmap showing the pairwise correlation between numerical features.")
    
    if corr_matrix is not None:
        # Mask upper triangle to avoid redundancy
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_display = corr_matrix.mask(mask)

        fig = px.imshow(
            corr_display,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu,
            title="Correlation Heatmap (Numerical Features)",
            height=600,
            width=800
        )
        # Configure layout to hide masked cells
        fig.update_layout(
            xaxis=dict(tickangle=-45),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
    elif numerical_df.shape[1] <= 1:
        st.info("Need at least two numerical features to compute a correlation matrix.")
    else:
        st.info("No numerical features available for correlation analysis.")
        
    # New Section: AI Analysis 
    st.header("5. AI-Driven Data Analysis Report")
    st.markdown("Generate a comprehensive, text-based analysis of the dataset, including interpretations of statistics and correlations, powered by a Large Language Model.")

    # State management for the generated report
    if 'ai_report_text' not in st.session_state:
        st.session_state.ai_report_text = None
        
    col_ai_btn, col_ai_placeholder = st.columns([1, 2])
    
    with col_ai_btn:
        if st.button("Generate AI Report", type="primary"):
            report_text = generate_ai_report(df)
            st.session_state.ai_report_text = report_text
            st.rerun() # Rerun to display the output properly
    
    if st.session_state.ai_report_text:
        # Display the generated report
        st.markdown("---")
        st.subheader("Generated Analysis")
        st.markdown(st.session_state.ai_report_text)
        st.markdown("---")

        # Make the report downloadable
        st.download_button(
            label="⬇️ Download AI Report (.txt)",
            data=st.session_state.ai_report_text,
            file_name="ai_data_analysis_report.txt",
            mime="text/plain",
            help="Download the full AI-generated text report."
        )


# Main Application Logic
def main():
    st.title("Dataset Analysis and Reporting Tool")
    st.caption("Upload a CSV file to generate a comprehensive, interactive data report.")
    st.markdown("_Now with AI-Powered text analysis and downloadable reports!_")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file here", 
        type=['csv']
    )

    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)

        if df is not None:
            st.success(f"Successfully loaded '{uploaded_file.name}'")
            st.divider()
            # Generate the report
            generate_report(df)
    else:
        st.info("Awaiting file upload. Please select a CSV file to begin analysis.")
        
        # Example dataset suggestion for demonstration
        st.markdown(
            """
            ### Quick Start Tip
            If you want to test the app, you can easily find sample CSV datasets 
            online, such as the famous Iris dataset or Titanic dataset.
            """
        )

if __name__ == '__main__':
    main()