import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Generate sample data
def generate_sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return pd.DataFrame({
        'Feature1': X[:, 0],
        'Feature2': X[:, 1],
        'Target': y
    })

# Create the layout
layout = html.Div(
    style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'},
    children=[
        html.Div(
            style={'padding': '20px', 'margin': '10px', 'borderRadius': '12px', 
                  'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                  'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
            children=[
                html.H1('ML Predictions Dashboard', 
                       style={'color': '#1a237e', 'fontWeight': '600', 
                             'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(
                    style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center'},
                    children=[
                        dcc.Dropdown(
                            id='ml-model-dropdown',
                            options=[
                                {'label': 'Linear Regression', 'value': 'linear'},
                                {'label': 'Logistic Regression', 'value': 'logistic'},
                                {'label': 'Random Forest', 'value': 'random_forest'},
                                {'label': 'SVM', 'value': 'svm'},
                                {'label': 'Neural Network', 'value': 'neural_net'}
                            ],
                            value='linear',
                            style={'width': '200px', 'backgroundColor': '#ffffff'}
                        ),
                        html.Button('Run Prediction', id='ml-run-button', 
                                  style={'backgroundColor': '#1a237e', 'color': 'white', 
                                        'border': 'none', 'padding': '10px 20px', 
                                        'borderRadius': '5px', 'cursor': 'pointer',
                                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                        'transition': 'all 0.3s ease'})
                    ]
                )
            ]
        ),
        
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'padding': '20px'},
            children=[
                # Data Distribution
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Data Distribution', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='ml-distribution-graph',
                                style={'height': '400px'})
                    ]
                ),
                
                # Model Performance
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Model Performance', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='ml-performance-graph',
                                style={'height': '400px'})
                    ]
                ),
                
                # Confusion Matrix
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Confusion Matrix', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='ml-confusion-matrix',
                                style={'height': '400px'})
                    ]
                ),
                
                # Learning Curve
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Learning Curve', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='ml-learning-curve',
                                style={'height': '400px'})
                    ]
                )
            ]
        )
    ]
)

# Set the app's layout
app.layout = layout

# Callbacks
@app.callback(
    [Output('ml-distribution-graph', 'figure'),
     Output('ml-performance-graph', 'figure'),
     Output('ml-confusion-matrix', 'figure'),
     Output('ml-learning-curve', 'figure')],
    [Input('ml-run-button', 'n_clicks')],
    [Input('ml-model-dropdown', 'value')]
)
def update_graphs(n_clicks, selected_model):
    if n_clicks is None:
        n_clicks = 0
    
    df = generate_sample_data()
    
    # Update distribution graph
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Scatter(
        x=df['Feature1'],
        y=df['Feature2'],
        mode='markers',
        marker=dict(
            color=df['Target'],
            colorscale='Viridis',
            size=10,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Data Points'
    ))
    dist_fig.update_layout(
        title='Feature Distribution',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update performance graph
    performance_fig = go.Figure()
    performance_fig.add_trace(go.Bar(
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y=[0.85, 0.82, 0.88, 0.85],
        marker_color='#3498db',
        marker_line_color='#1a237e',
        marker_line_width=1.5,
        opacity=0.8
    ))
    performance_fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    # Update confusion matrix
    confusion_fig = go.Figure(data=go.Heatmap(
        z=[[45, 5], [3, 47]],
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title='Count',
            titleside='right'
        )
    ))
    confusion_fig.update_layout(
        title='Confusion Matrix',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e')
    )
    
    # Update learning curve
    learning_fig = go.Figure()
    learning_fig.add_trace(go.Scatter(
        x=list(range(10, 100, 10)),
        y=[0.65, 0.72, 0.78, 0.82, 0.84, 0.85, 0.86, 0.86, 0.87],
        mode='lines+markers',
        name='Training Score',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8, color='#1a237e')
    ))
    learning_fig.add_trace(go.Scatter(
        x=list(range(10, 100, 10)),
        y=[0.60, 0.68, 0.75, 0.80, 0.82, 0.83, 0.84, 0.84, 0.85],
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8, color='#c0392b')
    ))
    learning_fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training Examples',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return dist_fig, performance_fig, confusion_fig, learning_fig

if __name__ == '__main__':
    app.run(debug=True) 