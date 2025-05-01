import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Generate sample data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    sales = np.random.normal(1000, 200, len(dates)) + np.linspace(0, 500, len(dates))
    return pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'Units': np.random.randint(50, 150, len(dates)),
        'Profit': sales * np.random.uniform(0.2, 0.4, len(dates))
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
                html.H1('Sales Analysis Dashboard', 
                       style={'color': '#1a237e', 'fontWeight': '600', 
                             'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(
                    style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center'},
                    children=[
                        dcc.Dropdown(
                            id='sales-metric-dropdown',
                            options=[
                                {'label': 'Sales Revenue', 'value': 'Sales'},
                                {'label': 'Units Sold', 'value': 'Units'},
                                {'label': 'Profit', 'value': 'Profit'}
                            ],
                            value='Sales',
                            style={'width': '200px', 'backgroundColor': '#ffffff'}
                        ),
                        dcc.Dropdown(
                            id='sales-timeframe-dropdown',
                            options=[
                                {'label': 'Daily', 'value': 'D'},
                                {'label': 'Weekly', 'value': 'W'},
                                {'label': 'Monthly', 'value': 'M'}
                            ],
                            value='D',
                            style={'width': '200px', 'backgroundColor': '#ffffff'}
                        )
                    ]
                )
            ]
        ),
        
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'padding': '20px'},
            children=[
                # Sales Trend
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Sales Trend', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='sales-trend-graph',
                                style={'height': '400px'})
                    ]
                ),
                
                # Distribution
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Distribution', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='sales-distribution-graph',
                                style={'height': '400px'})
                    ]
                ),
                
                # Summary Statistics
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Summary Statistics', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        html.Div(id='sales-summary-stats',
                                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'})
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
    [Output('sales-trend-graph', 'figure'),
     Output('sales-distribution-graph', 'figure'),
     Output('sales-summary-stats', 'children')],
    [Input('sales-metric-dropdown', 'value'),
     Input('sales-timeframe-dropdown', 'value')]
)
def update_graphs(selected_metric, selected_timeframe):
    df = generate_sample_data()
    
    if selected_timeframe != 'D':
        df = df.resample(selected_timeframe, on='Date').mean()
    
    # Update trend graph
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=df.index if selected_timeframe != 'D' else df['Date'],
        y=df[selected_metric],
        mode='lines+markers',
        name=selected_metric,
        line=dict(color='#3498db', width=2),
        marker=dict(size=8, color='#1a237e')
    ))
    trend_fig.update_layout(
        title=f'{selected_metric} Trend',
        xaxis_title='Date',
        yaxis_title=selected_metric,
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
    
    # Update distribution graph
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(
        x=df[selected_metric],
        nbinsx=30,
        marker_color='#3498db',
        opacity=0.7
    ))
    dist_fig.update_layout(
        title=f'{selected_metric} Distribution',
        xaxis_title=selected_metric,
        yaxis_title='Frequency',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    # Update summary statistics
    stats = df[selected_metric].describe()
    summary_stats = [
        html.Div(
            style={'flex': '1', 'minWidth': '200px', 'padding': '20px',
                  'backgroundColor': '#ffffff', 'borderRadius': '8px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'},
            children=[
                html.H3(name.title(), style={'color': '#1a237e', 'marginBottom': '10px'}),
                html.P(f"{value:,.2f}", style={'fontSize': '24px', 'color': '#3498db'})
            ]
        )
        for name, value in stats.items()
    ]
    
    return trend_fig, dist_fig, summary_stats

if __name__ == '__main__':
    app.run(debug=True) 