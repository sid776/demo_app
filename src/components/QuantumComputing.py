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
                html.H1('Quantum Computing Dashboard', 
                       style={'color': '#1a237e', 'fontWeight': '600', 
                             'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(
                    style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center'},
                    children=[
                        dcc.Dropdown(
                            id='qc-circuit-dropdown',
                            options=[
                                {'label': 'Quantum Fourier Transform', 'value': 'qft'},
                                {'label': 'Grover\'s Algorithm', 'value': 'grover'},
                                {'label': 'Quantum Teleportation', 'value': 'teleport'},
                                {'label': 'Quantum Entanglement', 'value': 'entangle'}
                            ],
                            value='qft',
                            style={'width': '200px', 'backgroundColor': '#ffffff'}
                        ),
                        html.Button('Run Circuit', id='qc-run-button', 
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
                # Quantum Circuit
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Quantum Circuit', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='qc-quantum-circuit',
                                style={'height': '400px'})
                    ]
                ),
                
                # State Vector
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('State Vector', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='qc-state-vector',
                                style={'height': '400px'})
                    ]
                ),
                
                # Measurement Results
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Measurement Results', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='qc-measurement-results',
                                style={'height': '400px'})
                    ]
                ),
                
                # Quantum Error
                html.Div(
                    style={'flex': '1', 'minWidth': '45%', 'padding': '20px', 'margin': '10px',
                          'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                          'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)'},
                    children=[
                        html.H2('Quantum Error', 
                               style={'color': '#1a237e', 'fontWeight': '600',
                                     'marginBottom': '20px'}),
                        dcc.Graph(id='qc-quantum-error',
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
    [Output('qc-quantum-circuit', 'figure'),
     Output('qc-state-vector', 'figure'),
     Output('qc-measurement-results', 'figure'),
     Output('qc-quantum-error', 'figure')],
    [Input('qc-run-button', 'n_clicks')],
    [Input('qc-circuit-dropdown', 'value')]
)
def update_graphs(n_clicks, selected_circuit):
    if n_clicks is None:
        n_clicks = 0
    
    # Update quantum circuit visualization
    circuit_fig = go.Figure()
    circuit_fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3],
        y=[0, 0, 0, 0],
        mode='lines+markers',
        name='Qubit 0',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8, color='#1a237e')
    ))
    circuit_fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3],
        y=[1, 1, 1, 1],
        mode='lines+markers',
        name='Qubit 1',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8, color='#c0392b')
    ))
    circuit_fig.update_layout(
        title='Quantum Circuit',
        xaxis_title='Time',
        yaxis_title='Qubit',
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
    
    # Update state vector visualization
    states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    probabilities = [0.25, 0.25, 0.25, 0.25]
    
    state_fig = go.Figure()
    state_fig.add_trace(go.Bar(
        x=states,
        y=probabilities,
        marker_color='#3498db',
        marker_line_color='#1a237e',
        marker_line_width=1.5,
        opacity=0.8
    ))
    state_fig.update_layout(
        title='State Vector',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    # Update measurement results
    measurements = ['000', '001', '010', '011', '100', '101', '110', '111']
    counts = [120, 80, 60, 40, 30, 20, 10, 5]
    
    measurement_fig = go.Figure()
    measurement_fig.add_trace(go.Bar(
        x=measurements,
        y=counts,
        marker_color='#3498db',
        marker_line_color='#1a237e',
        marker_line_width=1.5,
        opacity=0.8
    ))
    measurement_fig.update_layout(
        title='Measurement Results',
        yaxis_title='Count',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    # Update quantum error
    error_types = ['Bit Flip', 'Phase Flip', 'Depolarizing', 'Amplitude Damping']
    error_rates = [0.01, 0.02, 0.03, 0.04]
    
    error_fig = go.Figure()
    error_fig.add_trace(go.Bar(
        x=error_types,
        y=error_rates,
        marker_color='#3498db',
        marker_line_color='#1a237e',
        marker_line_width=1.5,
        opacity=0.8
    ))
    error_fig.update_layout(
        title='Quantum Error',
        yaxis_title='Error Rate',
        yaxis_range=[0, 0.05],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a237e'),
        showlegend=False
    )
    
    return circuit_fig, state_fig, measurement_fig, error_fig

if __name__ == '__main__':
    app.run(debug=True) 