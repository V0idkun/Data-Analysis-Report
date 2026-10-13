import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Data Analysis Report Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Function to generate the comprehensive analysis report
def generate_report(df, categorical_threshold=20):
    """Generates and displays the full data analysis report."""

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
        numerical_df = df.select_dtypes(include=np.number)
        if not numerical_df.empty:
            st.dataframe(numerical_df.describe().T, use_container_width=True)
        else:
            st.info("No numerical features found for summary statistics.")
            
    # 2.2 Missing Value Analysis
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

    # 3.1 Feature Iteration
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
    
    numerical_df = df.select_dtypes(include=np.number)
    if not numerical_df.empty and numerical_df.shape[1] > 1:
        corr_matrix = numerical_df.corr().round(2)
        
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


# --- Main Application Logic ---
def main():
    st.title("Dataset Analysis and Reporting Tool")
    st.caption("Upload a CSV file to generate a comprehensive, interactive data report.")

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