import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random

# ----------------------
# App Configuration
# ----------------------
st.set_page_config(
    page_title="Finance ML App",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# AF3005 – Programming for Finance | Assignment 3 | FAST-NUCES"
    }
)

# ----------------------
# Helper Functions
# ----------------------
GIFS = [
    "https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif",
    "https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif",
    "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
    "https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif",
    "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif"
]
def get_gif_html(gif_url, width="100%"):
    return f'<img src="{gif_url}" width="{width}" style="border-radius:10px;"/>'

def file_download(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results</a>'
    return href

def show_feature_importance(model, X):
    if hasattr(model, 'coef_'):
        importance = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        fig = px.bar(x=X.columns, y=importance, labels={'x':'Feature','y':'Importance'}, title="Feature Importance")
        st.plotly_chart(fig)

# ----------------------
# Sidebar
# ----------------------
selected_gif = random.choice(GIFS)
st.sidebar.markdown("# 📊 Finance ML App")
st.sidebar.markdown(get_gif_html(selected_gif, width="90%"), unsafe_allow_html=True)
st.sidebar.markdown("---")

# Step Progress
steps = [
    "Load Data", "Preprocessing", "Feature Engineering", "Train/Test Split", "Model Training", "Evaluation", "Results"
]
if 'step' not in st.session_state:
    st.session_state.step = 0
st.sidebar.markdown("## Step Progress")
for i, step in enumerate(steps):
    if i <= st.session_state.step:
        st.sidebar.success(f"{i+1}. {step}")
    else:
        st.sidebar.info(f"{i+1}. {step}")

# Help/Info Section
with st.sidebar.expander("ℹ️ Help / Info"):
    st.markdown("""
    - **Upload** a Kragle or Kaggle CSV dataset, or fetch data from Yahoo Finance.
    - **Navigate** each ML step using the buttons.
    - **Select** features and target for modeling.
    - **Download** results after evaluation.
    """)
    st.markdown("[Kaggle Datasets](https://www.kaggle.com/datasets)")

# Data upload/fetch
uploaded_file = st.sidebar.file_uploader("Upload Kragle/Kaggle Dataset (CSV)", type=["csv"])
ticker = st.sidebar.text_input("Or enter a Yahoo Finance Ticker (e.g. AAPL)")
fetch_btn = st.sidebar.button("Fetch Yahoo Finance Data")
example_btn = st.sidebar.button("Show Example Data")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed for AF3005 – Programming for Finance**")
st.sidebar.markdown("Instructor: Dr. Usama Arshad")

# ----------------------
# Main App
# ----------------------
st.markdown(
    f"""
    <div style='background-color:#0e1117;padding:20px;border-radius:10px;'>
        <h1 style='color:#f9d342;'>💹 Welcome to the Finance ML App!</h1>
        <p style='color:#f1f1f1;font-size:18px;'>
            This interactive app lets you explore financial data, apply machine learning models, and visualize results step by step.<br>
            <b>Upload your dataset or fetch real-time data, then walk through the ML pipeline interactively!</b>
        </p>
        <img src='{selected_gif}' width='300'>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state for workflow
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None

# Step 1: Load Data
st.header("Step 1: Load Data")
load_btn = st.button("Load Data")
if load_btn or example_btn:
    with st.spinner("Loading data..."):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("Dataset loaded successfully!")
            st.session_state.step = 1
        elif fetch_btn and ticker:
            try:
                df = yf.download(ticker, period="1y")
                if not df.empty:
                    st.session_state.df = df.reset_index()
                    st.success(f"Yahoo Finance data for {ticker} loaded!")
                    st.session_state.step = 1
                else:
                    st.error("No data found for the given ticker.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        elif example_btn:
            df = pd.DataFrame({
                'Open': np.random.rand(100)*100,
                'High': np.random.rand(100)*100,
                'Low': np.random.rand(100)*100,
                'Close': np.random.rand(100)*100,
                'Volume': np.random.randint(1000, 10000, 100)
            })
            st.session_state.df = df
            st.info("Example data loaded.")
            st.session_state.step = 1
        else:
            st.warning("Please upload a dataset or enter a ticker and click fetch.")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head())
        st.markdown(f"**Rows:** {st.session_state.df.shape[0]} | **Columns:** {st.session_state.df.shape[1]}")
        st.plotly_chart(px.histogram(st.session_state.df, x=st.session_state.df.columns[0]))

# Only proceed if data is loaded
if st.session_state.df is not None:
    # Step 2: Preprocessing
    st.header("Step 2: Preprocessing")
    preprocess_btn = st.button("Preprocess Data")
    if preprocess_btn:
        with st.spinner("Preprocessing data..."):
            df = st.session_state.df.copy()
            missing = df.isnull().sum().sum()
            st.info(f"Missing values in dataset: {missing}")
            df = df.dropna()
            st.session_state.df = df
            st.success("Missing values removed. Data cleaned.")
            st.dataframe(df.head())
            st.session_state.step = 2
            st.plotly_chart(px.imshow(df.corr(), text_auto=True, color_continuous_scale='blues', title="Correlation Heatmap"))

    # Step 3: Feature Engineering
    st.header("Step 3: Feature Engineering")
    feature_btn = st.button("Feature Engineering")
    if feature_btn:
        df = st.session_state.df.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for feature selection.")
        else:
            features = st.multiselect("Select features for modeling", numeric_cols, default=numeric_cols[:-1])
            target = st.selectbox("Select target variable", numeric_cols, index=len(numeric_cols)-1)
            st.session_state.features = features
            st.session_state.target = target
            st.info(f"Selected features: {features}")
            st.info(f"Target variable: {target}")
            st.success("Feature selection complete.")
            st.session_state.step = 3

    # Step 4: Train/Test Split
    st.header("Step 4: Train/Test Split")
    split_btn = st.button("Split Data")
    if split_btn:
        df = st.session_state.df.copy()
        features = st.session_state.features
        target = st.session_state.target
        if not features or not target:
            st.error("Please select features and target variable first.")
        else:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success("Data split into train and test sets.")
            pie = go.Figure(data=[go.Pie(labels=['Train', 'Test'], values=[len(X_train), len(X_test)], hole=.3)])
            st.plotly_chart(pie)
            st.session_state.step = 4

    # Step 5: Model Training
    st.header("Step 5: Model Training")
    model_type = st.selectbox("Choose ML Model", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])
    train_btn = st.button("Train Model")
    if train_btn:
        with st.spinner("Training model..."):
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.success("Linear Regression model trained.")
            elif model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.success("Logistic Regression model trained.")
            elif model_type == "K-Means Clustering":
                model = KMeans(n_clusters=3)
                model.fit(X_train)
                st.session_state.model = model
                st.success("K-Means clustering model trained.")
            st.session_state.step = 5

    # Step 6: Evaluation
    st.header("Step 6: Evaluation")
    eval_btn = st.button("Evaluate Model")
    if eval_btn:
        with st.spinner("Evaluating model..."):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            if model_type == "Linear Regression":
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                st.metric("Mean Squared Error", f"{mse:.2f}")
                fig = px.scatter(x=y_test, y=preds, labels={'x':'Actual','y':'Predicted'}, title="Actual vs Predicted")
                st.plotly_chart(fig)
                show_feature_importance(model, st.session_state.X_train)
                st.session_state.results = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
            elif model_type == "Logistic Regression":
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.metric("Accuracy", f"{acc*100:.2f}%")
                st.text(classification_report(y_test, preds))
                show_feature_importance(model, st.session_state.X_train)
                st.session_state.results = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
            elif model_type == "K-Means Clustering":
                X = st.session_state.X_train
                labels = model.labels_
                fig = px.scatter_matrix(X, color=labels, title="K-Means Clusters")
                st.plotly_chart(fig)
                st.session_state.results = pd.DataFrame(X)
                st.session_state.results['Cluster'] = labels
            st.success("Evaluation complete.")
            st.session_state.step = 6

    # Step 7: Results Visualization & Download
    st.header("Step 7: Results Visualization & Download")
    if st.session_state.results is not None:
        st.dataframe(st.session_state.results.head())
        st.markdown(file_download(st.session_state.results), unsafe_allow_html=True)
        st.balloons()
        st.markdown(get_gif_html(random.choice(GIFS), width="40%"), unsafe_allow_html=True)
        st.success("All steps complete! Download your results above.")

# ----------------------
# How to Run Note
# ----------------------
st.markdown("""
---
### How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Open the local URL in your browser.

---
**Course:** AF3005 – Programming for Finance  
**Instructor:** Dr. Usama Arshad  
**App by:** [Your Name]
""") 