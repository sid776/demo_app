import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Create the layout
layout = html.Div(
    style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'},
    children=[
        html.Div(
            style={'padding': '20px', 'margin': '10px', 'borderRadius': '12px', 
                  'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                  'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
            children=[
                html.H1('Deep Learning Dashboard', 
                       style={'color': '#1a237e', 'fontWeight': '600', 
                             'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(
                    style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center'},
                    children=[
                        dcc.Dropdown(
                            id='dl-model-dropdown',
                            options=[
                                {'label': 'CNN', 'value': 'cnn'},
                                {'label': 'RNN', 'value': 'rnn'},
                                {'label': 'Transformer', 'value': 'transformer'},
                                {'label': 'GAN', 'value': 'gan'}
                            ],
                            value='cnn',
                            style={'width': '200px', 'backgroundColor': '#ffffff'}
                        ),
                        html.Button('Train Model', id='dl-train-button', 
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
                # Training Progress
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Training Progress', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='dl-training-progress',
                                style={'height': '400px'})
                    ]
                ),
                
                # Model Architecture
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Model Architecture', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='dl-model-architecture',
                                style={'height': '400px'})
                    ]
                ),
                
                # Feature Maps
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Feature Maps', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='dl-feature-maps',
                                style={'height': '400px'})
                    ]
                ),
                
                # Performance Metrics
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Performance Metrics', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='dl-performance-metrics',
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
    [Output('dl-training-progress', 'figure'),
     Output('dl-model-architecture', 'figure'),
     Output('dl-feature-maps', 'figure'),
     Output('dl-performance-metrics', 'figure')],
    [Input('dl-train-button', 'n_clicks')],
    [Input('dl-model-dropdown', 'value')]
)
def update_graphs(n_clicks, selected_model):
    if n_clicks is None:
        n_clicks = 0
    
    # Update training progress
    epochs = list(range(1, 11))
    train_loss = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08]
    val_loss = [1.0, 0.85, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18]
    
    progress_fig = go.Figure()
    progress_fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8, color='#1a237e')
    ))
    progress_fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8, color='#c0392b')
    ))
    progress_fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss',
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
    
    # Update model architecture
    layers = ['Input', 'Conv1', 'Pool1', 'Conv2', 'Pool2', 'Dense1', 'Output']
    params = [0, 320, 0, 18496, 0, 1024, 10]
    
    arch_fig = go.Figure()
    arch_fig.add_trace(go.Bar(
        x=layers,
        y=params,
        marker_color='#3498db',
        marker_line_color='#1a237e',
        marker_line_width=1.5,
        opacity=0.8
    ))
    arch_fig.update_layout(
        title='Model Architecture',
        xaxis_title='Layer',
        yaxis_title='Parameters',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    # Update feature maps
    feature_maps = np.random.rand(8, 8, 3)
    feature_fig = go.Figure(data=go.Heatmap(
        z=feature_maps,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title='Intensity',
            titleside='right'
        )
    ))
    feature_fig.update_layout(
        title='Feature Maps',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e')
    )
    
    # Update performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [0.92, 0.91, 0.93, 0.92]
    
    metrics_fig = go.Figure()
    metrics_fig.add_trace(go.Bar(
        x=metrics,
        y=values,
        marker_color='#3498db',
        marker_line_color='#1a237e',
        marker_line_width=1.5,
        opacity=0.8
    ))
    metrics_fig.update_layout(
        title='Performance Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    return progress_fig, arch_fig, feature_fig, metrics_fig

if __name__ == '__main__':
    app.run(debug=True) 