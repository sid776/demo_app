import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
import requests
import seaborn as sns
from scipy import stats
import torch
import torch.nn as nn
import io
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.layers import Dropout, BatchNormalization
from qiskit.circuit.library import RealAmplitudes
import plotly.colors as pc

# Page config
st.set_page_config(
    page_title="Advanced Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Advanced Analytics Dashboard")
st.markdown("### Exploring Data Engineering, ML, Deep Learning, and Quantum AI")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sales Analysis", "ML Predictions", "Deep Learning", "Quantum Computing"])

# Enhanced color schemes
color_schemes = {
    'vibrant': px.colors.qualitative.Set3,
    'sequential': px.colors.sequential.Viridis,
    'diverging': px.colors.diverging.RdYlBu,
    'pastel': px.colors.qualitative.Pastel,
    'dark': px.colors.qualitative.Dark24
}

# Load multiple datasets
@st.cache_data
def load_datasets():
    # Original retail data
    retail_url = "https://raw.githubusercontent.com/microsoft/sql-server-samples/master/samples/databases/adventureworks/oltp-install-script/SalesOrderDetail_data.csv"
    
    try:
        df_retail = pd.read_csv(retail_url)
        df_retail['OrderDate'] = pd.date_range(start='2022-01-01', periods=len(df_retail), freq='H')
        df_retail['Revenue'] = df_retail['UnitPrice'] * df_retail['OrderQty']
        df_retail['Month'] = df_retail['OrderDate'].dt.to_period('M')
    except:
        # Generate synthetic retail data if URL fails
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        df_retail = pd.DataFrame({
            'OrderDate': dates,
            'Revenue': np.random.normal(1000, 200, len(dates)),
            'OrderQty': np.random.randint(1, 100, len(dates)),
            'UnitPrice': np.random.uniform(10, 1000, len(dates))
        })
        df_retail['Month'] = df_retail['OrderDate'].dt.to_period('M')
    
    # Generate synthetic customer behavior data
    n_customers = 1000
    df_customers = pd.DataFrame({
        'CustomerID': range(n_customers),
        'Age': np.random.normal(40, 15, n_customers),
        'Income': np.random.normal(50000, 20000, n_customers),
        'PurchaseFrequency': np.random.normal(10, 5, n_customers),
        'LoyaltyScore': np.random.uniform(0, 100, n_customers),
        'WebsiteVisits': np.random.poisson(15, n_customers),
        'CartAbandonment': np.random.binomial(1, 0.3, n_customers)
    })
    
    # Generate time series data
    timestamps = pd.date_range(start='2022-01-01', periods=1000, freq='H')
    df_timeseries = pd.DataFrame({
        'Timestamp': timestamps,
        'Temperature': np.sin(np.linspace(0, 50, 1000)) * 10 + 20 + np.random.normal(0, 2, 1000),
        'Humidity': np.cos(np.linspace(0, 50, 1000)) * 20 + 50 + np.random.normal(0, 5, 1000),
        'Energy': np.random.normal(1000, 100, 1000) + np.sin(np.linspace(0, 100, 1000)) * 200
    })
    
    return df_retail, df_customers, df_timeseries

# Load all datasets
df_retail, df_customers, df_timeseries = load_datasets()

if page == "Sales Analysis":
    st.header("Understanding Our Sales Trends")
    
    # Add explanation column
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Time Series Decomposition")
        # Convert to time series
        ts_data = df_retail.groupby('OrderDate')['Revenue'].sum()
        
        # Calculate components
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(ts_data, period=30)
        
        # Plot components
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Original', line=dict(color=color_schemes['vibrant'][0])))
        fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, name='Trend', line=dict(color=color_schemes['vibrant'][1])))
        fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, name='Seasonal', line=dict(color=color_schemes['vibrant'][2])))
        fig.update_layout(title='Time Series Components')
        st.plotly_chart(fig)
    
    with col2:
        st.markdown("""
        **Time Series Decomposition Explained:**
        
        This analysis breaks down the revenue time series into three components:
        1. **Trend**: Long-term progression
        2. **Seasonal**: Repeating patterns
        3. **Residual**: Random variations
        
        Understanding these components helps in:
        - Identifying long-term growth/decline
        - Planning for seasonal variations
        - Detecting anomalies
        """)

    # Add more visualizations with explanations...
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("2. Customer Purchase Patterns")
        # Create heatmap of purchase patterns
        hourly_data = df_retail.groupby([df_retail['OrderDate'].dt.hour, df_retail['OrderDate'].dt.dayofweek])['Revenue'].mean().unstack()
        
        fig = px.imshow(hourly_data,
                       labels=dict(x="Day of Week", y="Hour of Day", color="Average Revenue"),
                       color_continuous_scale=color_schemes['sequential'])
        fig.update_layout(title='Purchase Pattern Heatmap')
        st.plotly_chart(fig)
    
    with col4:
        st.markdown("""
        **Purchase Pattern Analysis:**
        
        The heatmap reveals:
        - Peak shopping hours
        - Weekly patterns
        - Customer behavior trends
        
        This helps in:
        - Optimizing staffing
        - Planning promotions
        - Inventory management
        """)

elif page == "ML Predictions":
    st.header("Predicting Sales with Smart Tools")
    st.markdown("See how different smart models help us forecast sales and understand our customers. Each chart gives you a business takeaway.")

    # Prepare data
    X = df_retail[['OrderQty', 'UnitPrice']].values
    y = df_retail['Revenue'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressors = {
        'Linear Regression': lambda: __import__('sklearn.linear_model').linear_model.LinearRegression(),
        'Decision Tree': lambda: DecisionTreeRegressor(),
        'Random Forest': lambda: RandomForestRegressor(n_estimators=100),
        'Gradient Boosting': lambda: GradientBoostingRegressor(),
        'KNN Regression': lambda: __import__('sklearn.neighbors').neighbors.KNeighborsRegressor(),
        'SVR': lambda: __import__('sklearn.svm').svm.SVR(),
        'Ridge Regression': lambda: __import__('sklearn.linear_model').linear_model.Ridge(),
        'Lasso Regression': lambda: __import__('sklearn.linear_model').linear_model.Lasso(),
    }
    metrics = []
    color_list = color_schemes['vibrant']
    for i, (name, reg_fn) in enumerate(regressors.items()):
        model = reg_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        mae = __import__('sklearn.metrics').metrics.mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(__import__('sklearn.metrics').metrics.mean_squared_error(y_test, y_pred))
        metrics.append({'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
        # Visual: Actual vs Predicted
        fig = px.scatter(x=y_test, y=y_pred, title=f"{name}: Actual vs Predicted", color_discrete_sequence=[color_list[i%len(color_list)]])
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction', line=dict(color='black', dash='dash')))
        st.plotly_chart(fig)
        st.markdown(f"""
        This chart shows how well our smart sales tool can estimate revenue based on order size and price. When the dots are close to the line, it means our predictions are very close to what actually happened. For example, with this method, our average error is just {rmse:.2f} and we explain about {r2:.2f} of the ups and downs in revenue. This helps us plan inventory and set targets with more confidence.
        """)

    # Clustering
    st.subheader("Clustering Techniques")
    X_cluster = df_customers[['Income', 'PurchaseFrequency']].values
    kmeans = KMeans(n_clusters=4)
    clusters = kmeans.fit_predict(X_cluster)
    sil_score = __import__('sklearn.metrics').metrics.silhouette_score(X_cluster, clusters)
    fig = px.scatter(x=X_cluster[:, 0], y=X_cluster[:, 1], color=clusters, title="KMeans Clustering", color_continuous_scale=color_schemes['sequential'])
    st.plotly_chart(fig)
    st.markdown(f"""
    Here, we've grouped our customers by how much they spend and how often they buy. Each color is a different type of customer. This makes it easy to spot our best buyers and those who might need a nudge. The clearer the groups, the more targeted our campaigns can be.
    """)

    dbscan = DBSCAN(eps=5000, min_samples=5)
    db_clusters = dbscan.fit_predict(X_cluster)
    sil_score_db = sil_score if len(set(db_clusters)) > 1 else float('nan')
    fig = px.scatter(x=X_cluster[:, 0], y=X_cluster[:, 1], color=db_clusters, title="DBSCAN Clustering", color_continuous_scale=color_schemes['diverging'])
    st.plotly_chart(fig)
    st.markdown(f"""
    This chart helps us find unusual customer groups and outliers. Some customers behave differently—maybe they're big spenders or only buy once. Spotting these helps us tailor our approach and catch surprises early.
    """)

    # Dimensionality Reduction
    st.subheader("Dimensionality Reduction")
    X_pca = df_customers[['Age', 'Income', 'PurchaseFrequency', 'LoyaltyScore']].values
    pca = PCA(n_components=2)
    X_pca_reduced = pca.fit_transform(X_pca)
    fig = px.scatter(x=X_pca_reduced[:, 0], y=X_pca_reduced[:, 1], title="PCA: Customer Data in 2D", color=df_customers['LoyaltyScore'], color_continuous_scale=color_schemes['sequential'])
    st.plotly_chart(fig)
    st.markdown("""
    We've taken lots of customer details and turned them into a simple map. Customers who are close together on this map behave similarly. This helps us quickly spot patterns and find groups to focus our marketing on.
    """)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], title="t-SNE: Customer Data", color=df_customers['LoyaltyScore'], color_continuous_scale=color_schemes['diverging'])
    st.plotly_chart(fig)
    st.markdown("""
    This chart is like a magic map that shows hidden groups in your customer data. It helps us see which customers are similar, so we can design better offers and campaigns.
    """)

    # Anomaly Detection
    st.subheader("Anomaly Detection")
    X_anomaly = df_retail[['Revenue', 'OrderQty']].values
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X_anomaly)
    fig = px.scatter(x=X_anomaly[:, 0], y=X_anomaly[:, 1], color=anomalies, color_discrete_map={1: color_schemes['vibrant'][0], -1: color_schemes['vibrant'][1]}, title='Isolation Forest Anomaly Detection')
    st.plotly_chart(fig)
    st.markdown("""
    This chart highlights unusual orders that stand out from the rest. These could be big opportunities, mistakes, or even fraud. By catching these early, we can act fast and avoid surprises.
    """)

    # Summary Table
    st.subheader("ML Model Comparison Table")
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    fig = px.bar(metrics_df, x='Model', y=['R2', 'MAE', 'RMSE'], barmode='group', title='ML Model Metrics Comparison', color_discrete_sequence=color_schemes['vibrant'])
    st.plotly_chart(fig)
    st.markdown("**Statistical Superiority:** Models like Random Forest and Gradient Boosting often outperform Linear Regression in R² and RMSE, especially on nonlinear data.")

elif page == "Deep Learning":
    st.header("Smarter Customer Insights with AI")
    st.markdown("Our AI models help us spot loyal customers, predict future sales, and find hidden patterns. Each chart below shows what this means for your team.")

    # Prepare data
    X = df_customers[['Age', 'Income', 'PurchaseFrequency']].values
    y = df_customers['LoyaltyScore'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dl_metrics = []
    color_list = color_schemes['vibrant']

    # 1. Feedforward Neural Network
    with st.container():
        st.subheader("1. Feedforward Neural Network")
        model = Sequential([
            Dense(64, activation='relu', input_shape=(3,)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_scaled, y_train, epochs=30, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_scaled).flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        dl_metrics.append({'Model': 'Feedforward NN', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=[color_list[0]], title="Feedforward NN: Actual vs Predicted")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past customer data to predict loyalty scores. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can identify which customers are most likely to return, helping us focus our retention efforts.
        """)

    # 2. Deep MLP
    with st.container():
        st.subheader("2. Deep MLP (Multi-layer Perceptron)")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(3,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_scaled, y_train, epochs=30, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_scaled).flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        dl_metrics.append({'Model': 'Deep MLP', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=[color_list[1]], title="Deep MLP: Actual vs Predicted")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past customer data to predict loyalty scores. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can identify which customers are most likely to return, helping us focus our retention efforts.
        """)

    # 3. LSTM (on time series)
    with st.container():
        st.subheader("3. LSTM (Long Short-Term Memory)")
        # Use time series data
        seq_length = 10
        def create_sequences(data, seq_length):
            sequences = []
            targets = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:(i + seq_length)])
                targets.append(data[i + seq_length])
            return np.array(sequences), np.array(targets)
        X_seq, y_seq = create_sequences(df_timeseries['Energy'].values, seq_length)
        X_train_seq = X_seq[:800]
        X_test_seq = X_seq[800:]
        y_train_seq = y_seq[:800]
        y_test_seq = y_seq[800:]
        model = Sequential([
            LSTM(32, input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_seq.reshape(-1, seq_length, 1), y_train_seq, epochs=20, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_seq.reshape(-1, seq_length, 1)).flatten()
        rmse = np.sqrt(np.mean((y_test_seq - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test_seq, y_pred)
        dl_metrics.append({'Model': 'LSTM', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test_seq, y=y_pred, color_discrete_sequence=[color_list[2]], title="LSTM: Actual vs Predicted (Energy)")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past energy consumption data to predict future consumption. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can plan energy usage and avoid surprises.
        """)

    # 4. 1D CNN (on time series)
    with st.container():
        st.subheader("4. 1D CNN (Convolutional Neural Network)")
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(seq_length, 1)),
            MaxPooling1D(2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_seq.reshape(-1, seq_length, 1), y_train_seq, epochs=20, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_seq.reshape(-1, seq_length, 1)).flatten()
        rmse = np.sqrt(np.mean((y_test_seq - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test_seq, y_pred)
        dl_metrics.append({'Model': '1D CNN', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test_seq, y=y_pred, color_discrete_sequence=[color_list[3]], title="1D CNN: Actual vs Predicted (Energy)")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past energy consumption data to predict future consumption. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can plan energy usage and avoid surprises.
        """)

    # 5. Autoencoder
    with st.container():
        st.subheader("5. Autoencoder for Dimensionality Reduction")
        input_dim = X_train_scaled.shape[1]
        encoding_dim = 2
        autoencoder = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(encoding_dim, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])
        autoencoder.compile(optimizer='adam', loss='mse')
        ae_history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=30, validation_split=0.2, verbose=0)
        encoder = Sequential(autoencoder.layers[:3])
        encoded_data = encoder.predict(X_train_scaled)
        fig = px.scatter(x=encoded_data[:, 0], y=encoded_data[:, 1], color=y_train, color_continuous_scale=color_schemes['sequential'], title='Autoencoder: 2D Representation of Customer Data')
        st.plotly_chart(fig)
        recon_loss = ae_history.history['val_loss'][-1]
        st.markdown(f"""
        This chart shows how well our AI model can predict customer loyalty scores. The closer the dots are to the line, the better our predictions. With a reconstruction loss of {recon_loss:.4f}, we can identify anomalies and tailor our approach.
        """)

    # 6. Dropout Variant
    with st.container():
        st.subheader("6. MLP with Dropout")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(3,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_scaled, y_train, epochs=30, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_scaled).flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        dl_metrics.append({'Model': 'MLP+Dropout', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=[color_list[4]], title="MLP+Dropout: Actual vs Predicted")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past customer data to predict loyalty scores. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can identify which customers are most likely to return, helping us focus our retention efforts.
        """)

    # 7. BatchNorm Variant
    with st.container():
        st.subheader("7. MLP with BatchNorm")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(3,)),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_scaled, y_train, epochs=30, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_scaled).flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        dl_metrics.append({'Model': 'MLP+BatchNorm', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=[color_list[5]], title="MLP+BatchNorm: Actual vs Predicted")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past customer data to predict loyalty scores. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can identify which customers are most likely to return, helping us focus our retention efforts.
        """)

    # 8. PyTorch NN
    with st.container():
        st.subheader("8. PyTorch Feedforward NN")
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        net = SimpleNN()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = net(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        y_pred = net(X_test_t).detach().numpy().flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        dl_metrics.append({'Model': 'PyTorch NN', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=[color_list[6]], title="PyTorch NN: Actual vs Predicted")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past customer data to predict loyalty scores. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can identify which customers are most likely to return, helping us focus our retention efforts.
        """)

    # 9. RNN (Simple Recurrent Neural Network)
    with st.container():
        st.subheader("9. Simple RNN (Recurrent Neural Network)")
        model = Sequential([
            tf.keras.layers.SimpleRNN(32, input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_seq.reshape(-1, seq_length, 1), y_train_seq, epochs=20, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_seq.reshape(-1, seq_length, 1)).flatten()
        rmse = np.sqrt(np.mean((y_test_seq - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test_seq, y_pred)
        dl_metrics.append({'Model': 'Simple RNN', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test_seq, y=y_pred, color_discrete_sequence=[color_list[7]], title="Simple RNN: Actual vs Predicted (Energy)")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past energy consumption data to predict future consumption. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can plan energy usage and avoid surprises.
        """)

    # 10. Deep Ensemble (Averaged Predictions)
    with st.container():
        st.subheader("10. Deep Ensemble (Averaged Predictions)")
        preds = []
        for _ in range(3):
            model = Sequential([
                Dense(64, activation='relu', input_shape=(3,)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=0)
            preds.append(model.predict(X_test_scaled).flatten())
        y_pred = np.mean(preds, axis=0)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = __import__('sklearn.metrics').metrics.r2_score(y_test, y_pred)
        dl_metrics.append({'Model': 'Deep Ensemble', 'RMSE': rmse, 'R2': r2})
        fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=[color_list[8]], title="Deep Ensemble: Actual vs Predicted")
        st.plotly_chart(fig)
        st.markdown(f"""
        Our AI model learns from past customer data to predict loyalty scores. The closer the dots are to the line, the better our predictions. With an average error of {rmse:.2f}, we can identify which customers are most likely to return, helping us focus our retention efforts.
        """)

    # Summary Table
    st.subheader("Deep Learning Model Comparison Table")
    dl_metrics_df = pd.DataFrame(dl_metrics)
    st.dataframe(dl_metrics_df)
    fig = px.bar(dl_metrics_df, x='Model', y=['RMSE', 'R2'], barmode='group', title='Deep Learning Model Metrics Comparison', color_discrete_sequence=color_list)
    st.plotly_chart(fig)
    st.markdown("**Statistical Superiority:** Deep models (LSTM, CNN, Ensemble) often outperform shallow models, especially on complex or sequential data.")

elif page == "Quantum Computing":
    st.header("Exploring the Future: Quantum Insights")
    st.markdown("Quantum technology is on the horizon. Here's how it could change the way we analyze data and keep our business secure.")
    
    # Add new quantum computing techniques...
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("4. Quantum Superposition States")
        # Create superposition circuit
        qc_super = QuantumCircuit(3, 3)
        qc_super.h(0)
        qc_super.h(1)
        qc_super.h(2)
        qc_super.measure_all()
        
        # Define backend for quantum simulation
        backend = Aer.get_backend('qasm_simulator')
        # Run on simulator
        job = backend.run(qc_super, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Plot with custom colors
        fig = go.Figure(data=[
            go.Bar(x=list(counts.keys()),
                  y=list(counts.values()),
                  marker_color=color_schemes['vibrant'])
        ])
        fig.update_layout(title='Quantum Superposition States Distribution')
        st.plotly_chart(fig)
        st.markdown("""
        This chart is a sneak peek into the future of analytics. Quantum computers can look at many possibilities at once, helping us solve problems faster and keep our data secure. While this is still experimental, it shows where technology is heading for business insights.
        """)

        # --- Quantum Parallelism Example ---
        st.subheader("5. Quantum Parallelism Example")
        st.markdown("This circuit demonstrates how quantum computers can evaluate a function on all possible inputs simultaneously.")
        qc_par = QuantumCircuit(2, 2)
        qc_par.h([0, 1])  # Put both qubits in superposition
        qc_par.measure([0, 1], [0, 1])
        job = backend.run(qc_par, shots=1024)
        result = job.result()
        counts = result.get_counts()
        fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                     title="Quantum Parallelism: All States Sampled",
                     labels={'x': 'State', 'y': 'Frequency'}, color_discrete_sequence=color_schemes['sequential'])
        st.plotly_chart(fig)
        st.markdown("""
        Quantum computers can try all possible answers at once. This could mean much faster analysis for things like customer segmentation or fraud detection in the future.
        """)

        # --- Quantum Search (Grover's Algorithm) ---
        st.subheader("6. Quantum Search (Grover's Algorithm)")
        st.markdown("Grover's algorithm finds a marked item in an unsorted database quadratically faster than classical search.")
        from qiskit.circuit.library import GroverOperator
        from qiskit.algorithms import AmplificationProblem, Grover
        from qiskit.quantum_info import Statevector
        # Simple oracle for |11> state
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        problem = AmplificationProblem(oracle=oracle, is_good_state=lambda x: x == '11')
        grover = Grover(quantum_instance=backend)
        result = grover.amplify(problem)
        # Visualize the result
        sv = Statevector.from_instruction(grover_op)
        probs = sv.probabilities_dict()
        fig = px.bar(x=list(probs.keys()), y=list(probs.values()),
                     title="Grover's Algorithm State Probabilities",
                     labels={'x': 'State', 'y': 'Probability'}, color_discrete_sequence=color_schemes['diverging'])
        st.plotly_chart(fig)
        st.markdown("""
        This chart shows how quantum computers could help us find the "needle in a haystack"—like a VIP customer or a rare event—much faster than today's computers.
        """)

        # --- Quantum Cryptography (BB84 Protocol) ---
        st.subheader("7. Quantum Cryptography (BB84 Protocol)")
        st.markdown("BB84 is a quantum key distribution protocol that uses quantum states to securely share a key.")
        # Simulate BB84 key distribution
        n_bits = 50
        alice_bits = np.random.randint(2, size=n_bits)
        alice_bases = np.random.randint(2, size=n_bits)
        bob_bases = np.random.randint(2, size=n_bits)
        # Bob's measurement results
        bob_results = [alice_bits[i] if alice_bases[i] == bob_bases[i] else np.random.randint(2) for i in range(n_bits)]
        # Key agreement
        key = [bob_results[i] for i in range(n_bits) if alice_bases[i] == bob_bases[i]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(len(key))), y=key, marker_color=color_schemes['dark']))
        fig.update_layout(title='BB84 Shared Key Bits', xaxis_title='Bit Index', yaxis_title='Bit Value')
        st.plotly_chart(fig)
        st.markdown("""
        Here's how quantum technology could keep our business secrets safe. This chart shows a secret code made using quantum physics—if anyone tries to listen in, we'll know!
        """)

    with col2:
        st.markdown("""
        **Quantum Superposition:**
        
        Key concepts:
        1. **Superposition**: Multiple states simultaneously
        2. **Hadamard Gates**: Creates equal superposition
        3. **Measurement**: Collapses to classical state
        
        Applications:
        - Quantum parallelism
        - Quantum search
        - Quantum cryptography
        
        ---
        
        **Quantum Parallelism:**
        Quantum computers can evaluate a function on all possible inputs at once, thanks to superposition. This is the basis for quantum speedup in algorithms like Grover's.
        
        ---
        
        **Grover's Algorithm:**
        Grover's search finds a marked item in an unsorted list in O(√N) time, compared to O(N) classically. The probability distribution shows amplification of the target state.
        
        ---
        
        **BB84 Protocol:**
        The BB84 protocol uses quantum states to securely share a cryptographic key. If an eavesdropper tries to intercept, the quantum state collapses, revealing the intrusion.
        """)

# Footer
st.markdown("---")
st.markdown("### About this Dashboard")
st.markdown("""
This dashboard demonstrates various advanced analytics techniques:
- **Sales Analysis**: Comprehensive visualization of sales data
- **ML Predictions**: Advanced machine learning algorithms and visualizations
- **Deep Learning**: Neural network architectures and training
- **Quantum Computing**: Quantum algorithms and simulations
""") 