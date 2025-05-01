import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Initialize the Dash app with custom styling
app = dash.Dash(__name__)

# Add external stylesheets for Google Fonts
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            * {
                font-family: 'Inter', sans-serif;
            }
            h1, h2, h3, h4, h5, h6, .header-text {
                font-family: 'Poppins', sans-serif;
            }
            
            /* Custom dropdown styling */
            .Select-control {
                border: 2px solid transparent !important;
                background: linear-gradient(#fff, #fff) padding-box,
                            linear-gradient(45deg, var(--primary-color), var(--secondary-color)) border-box !important;
                transition: all 0.3s ease !important;
            }
            
            .Select-control:hover {
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
                transform: translateY(-1px) !important;
            }
            
            .Select-menu-outer {
                border: none !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
                border-radius: 8px !important;
                margin-top: 8px !important;
                background: white !important;
            }
            
            .Select-option {
                padding: 12px !important;
                transition: all 0.2s ease !important;
            }
            
            .Select-option:hover {
                background-color: rgba(74, 144, 205, 0.1) !important;
            }
            
            .Select-option.is-selected {
                background-color: rgba(74, 144, 205, 0.2) !important;
                font-weight: 500 !important;
            }
            
            .Select-value {
                padding: 8px !important;
            }
            
            .Select-arrow-zone {
                padding-right: 12px !important;
            }
            
            .Select-placeholder {
                padding: 8px !important;
            }
        </style>
    </head>
    <body>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Set CSS variables for the gradient based on the current tab
                document.documentElement.style.setProperty('--primary-color', '#4A90CD');
                document.documentElement.style.setProperty('--secondary-color', '#00BBF9');
            });
        </script>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Custom color schemes
COLORS = {
    'background': '#FFFFFF',
    'secondary_background': '#F8FAFC',  # Very light blue-gray
    'text': '#4A90CD',  # Bright blue text
    # Tab-specific colors
    'sales': '#4A90CD',      # Bright blue
    'ml': '#9B5DE5',        # Bright purple
    'dl': '#F15BB5',        # Bright pink
    'qc': '#00BBF9',        # Sky blue
    # Additional colors
    'accent1': '#FEE440',    # Bright yellow
    'accent2': '#00F5D4',    # Bright turquoise
    'accent3': '#FF8FB1'     # Soft pink
}

GRAPH_THEME = {
    'paper_bgcolor': COLORS['background'],
    'plot_bgcolor': COLORS['secondary_background'],
    'font': {'color': COLORS['text']},
    'grid': {'color': '#E0E6ED'}
}

# Define algorithms for each domain
ML_ALGORITHMS = [
    'Random Forest',
    'Support Vector Machine',
    'Gradient Boosting',
    'K-Nearest Neighbors',
    'Neural Network'
]

DL_ALGORITHMS = [
    'Convolutional Neural Network',
    'Recurrent Neural Network',
    'Transformer',
    'Autoencoder',
    'GAN'
]

QC_ALGORITHMS = [
    'Quantum Fourier Transform',
    'Grover Search',
    'VQE',
    'QAOA',
    'Quantum Teleportation'
]

# Custom styles
CONTENT_STYLE = {
    'position': 'relative',
    'backgroundColor': COLORS['background'],
    'color': COLORS['text'],
    'padding': '20px',
    'borderRadius': '10px',
    'marginTop': '20px',
    'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.05)',
    'zIndex': '2'
}

HEADER_STYLE = {
    'position': 'fixed',
    'top': '20px',
    'left': '20px',
    'right': '20px',
    'backgroundColor': COLORS['sales'],
    'padding': '15px 20px',
    'color': COLORS['background'],
    'borderRadius': '10px',
    'textAlign': 'center',
    'fontSize': '2.2em',
    'fontWeight': '600',
    'marginBottom': '20px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
    'fontFamily': 'Poppins, sans-serif',
    'letterSpacing': '-0.5px',
    'zIndex': '1000'
}

LABEL_STYLE = {
    'color': COLORS['text'],
    'fontSize': '1.1em',
    'fontWeight': '500',
    'fontFamily': 'Inter, sans-serif',
    'marginBottom': '8px',
    'display': 'block'
}

# Update dropdown styles
DROPDOWN_STYLE = {
    'backgroundColor': COLORS['background'],
    'color': COLORS['text'],
    'border': 'none',  # Remove default border
    'borderRadius': '8px',
    'padding': '12px',
    'width': '100%',
    'font-size': '1.1em',
    'cursor': 'pointer',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
    'transition': 'all 0.3s ease',
}

DROPDOWN_CONTAINER_STYLE = {
    'backgroundColor': COLORS['background'],
    'padding': '2px',  # Reduced padding for the glow effect
    'borderRadius': '10px',
    'marginBottom': '25px',
    'width': '100%',
    'maxWidth': '500px',  # Limit maximum width
    'position': 'relative',
}

# Update tab styles
def get_tab_style(tab_value):
    return {
        'backgroundColor': COLORS['background'],
        'color': COLORS[tab_value],
        'padding': '12px 20px',
        'borderRadius': '8px 8px 0 0',
        'border': f'1px solid {COLORS[tab_value]}',
        'borderBottom': 'none',
        'marginRight': '2px',
        'fontSize': '1.05em',
        'fontFamily': 'Inter, sans-serif',
        'fontWeight': '500',
        'letterSpacing': '0.2px'
    }

def get_tab_selected_style(tab_value):
    return {
        'backgroundColor': COLORS[tab_value],
        'color': COLORS['background'],
        'padding': '12px 20px',
        'borderRadius': '8px 8px 0 0',
        'border': f'1px solid {COLORS[tab_value]}',
        'borderBottom': 'none',
        'marginRight': '2px',
        'fontWeight': '600',
        'fontSize': '1.05em',
        'fontFamily': 'Inter, sans-serif',
        'letterSpacing': '0.2px'
    }

SLIDER_STYLE = {
    'margin': '20px 0'
}

STATS_STYLE = {
    'backgroundColor': COLORS['background'],
    'padding': '20px',
    'borderRadius': '10px',
    'marginTop': '20px',
    'border': f'1px solid {COLORS["sales"]}',
    'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.05)'
}

# Update content styles for sidebar layout
SIDEBAR_STYLE = {
    'position': 'absolute',
    'top': '40px',  # Decreased from 45px
    'left': '20px',
    'width': '280px',
    'padding': '0 20px 20px 20px',  # Removed top padding
    'background-color': '#ffffff',
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'height': 'fit-content',
    'zIndex': '1'
}

MAIN_CONTENT_STYLE = {
    'position': 'relative',
    'marginLeft': '320px',
    'marginTop': '0',
    'padding': '0 20px 20px 20px',  # Removed top padding
    'backgroundColor': '#ffffff',
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'minHeight': '500px',
    'zIndex': '2'
}

# Update slider container style
SLIDER_CONTAINER_STYLE = {
    'backgroundColor': COLORS['background'],
    'padding': '15px',
    'borderRadius': '8px',
    'marginBottom': '20px',
    'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.05)',
    'border': '1px solid rgba(0, 0, 0, 0.05)'
}

# Update the graph container styles
GRAPH_CONTAINER_STYLE = {
    'marginBottom': '40px',
    'padding': '20px',
    'backgroundColor': '#ffffff',
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'position': 'relative',
    'zIndex': '2'
}

# Update the text container styles
TEXT_CONTAINER_STYLE = {
    'marginBottom': '30px',
    'paddingLeft': '20px',
    'paddingRight': '20px',
    'position': 'relative',
    'zIndex': '2'
}

app.layout = html.Div([
    html.Div('Advanced Analytics Dashboard', style=HEADER_STYLE),
    html.Div(style={'height': '80px'}),  # Spacer for fixed header
    
    html.Div([  # Added wrapper div for content positioning
        dcc.Tabs(id='tabs', value='ml', 
            style={
                'backgroundColor': COLORS['background'],
                'padding': '10px 10px 0 10px',
                'borderRadius': '8px',
                'position': 'relative',
                'zIndex': '3',
                'marginBottom': '5px'  # Decreased margin bottom for less spacing
            },
            colors={
                "border": COLORS['sales'],
                "primary": COLORS['sales'],
                "background": COLORS['background']
            },
            children=[
                dcc.Tab(label='Machine Learning Analysis', value='ml',
                        style=get_tab_style('ml'),
                        selected_style=get_tab_selected_style('ml')),
                dcc.Tab(label='Deep Learning Analysis', value='dl',
                        style=get_tab_style('dl'),
                        selected_style=get_tab_selected_style('dl')),
                dcc.Tab(label='Quantum Computing Analysis', value='qc',
                        style=get_tab_style('qc'),
                        selected_style=get_tab_selected_style('qc')),
                dcc.Tab(label='General Analysis', value='sales',
                        style=get_tab_style('sales'),
                        selected_style=get_tab_selected_style('sales')),
                dcc.Tab(label='README', value='readme',
                        style=get_tab_style('sales'),
                        selected_style=get_tab_selected_style('sales'))
            ]),
        
        html.Div(id='tabs-content', style=CONTENT_STYLE)
    ], style={'position': 'relative', 'minHeight': '100vh'})  # Added wrapper div style
], style={'backgroundColor': COLORS['secondary_background'], 'minHeight': '100vh', 'padding': '20px', 'paddingTop': '100px'})

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    tab_color = 'sales' if tab == 'readme' else tab
    color_script = html.Script(f'''
        document.documentElement.style.setProperty('--primary-color', '{COLORS[tab_color]}');
        document.documentElement.style.setProperty('--secondary-color', '{COLORS["accent2"]}');
    ''')
    
    if tab == 'readme':
        return html.Div([
            color_script,
            html.Div([
                html.H2('Dashboard Overview', style={'color': COLORS['sales'], 'marginBottom': '20px'}),
                html.P([
                    'This dashboard provides interactive visualizations and analysis tools across multiple domains. ',
                    'Each tab offers different analytical capabilities with adjustable parameters.'
                ], style={'fontSize': '1.1em', 'marginBottom': '30px'}),
                
                html.H3('Sales Analysis', style={'color': COLORS['sales'], 'marginBottom': '15px'}),
                html.P([
                    'Simulates sales data with adjustable mean and standard deviation. ',
                    'Visualizes sales trends over time and distribution patterns. ',
                    'Uses normally distributed random data to model realistic sales patterns.'
                ], style={'marginBottom': '25px'}),
                
                html.H3('Machine Learning', style={'color': COLORS['ml'], 'marginBottom': '15px'}),
                html.P([
                    'Demonstrates various ML algorithms with configurable hyperparameters. ',
                    'Shows training accuracy and loss curves. ',
                    'Algorithms include Random Forest, SVM, Gradient Boosting, KNN, and Neural Networks.'
                ], style={'marginBottom': '25px'}),
                
                html.H3('Deep Learning', style={'color': COLORS['dl'], 'marginBottom': '15px'}),
                html.P([
                    'Explores deep learning architectures with adjustable parameters. ',
                    'Visualizes training and validation accuracy. ',
                    'Includes CNN, RNN, Transformer, Autoencoder, and GAN architectures.'
                ], style={'marginBottom': '25px'}),
                
                html.H3('Quantum Computing', style={'color': COLORS['qc'], 'marginBottom': '15px'}),
                html.P([
                    'Simulates quantum algorithms with configurable circuit parameters. ',
                    'Shows quantum state probabilities and distributions. ',
                    'Features QFT, Grover Search, VQE, QAOA, and Quantum Teleportation.'
                ], style={'marginBottom': '25px'}),
                
                html.H3('Data Generation', style={'color': COLORS['sales'], 'marginBottom': '15px'}),
                html.P([
                    'All data in this dashboard is simulated for demonstration purposes. ',
                    'Sales data uses normal distribution with adjustable parameters. ',
                    'ML/DL metrics use exponential decay with noise to simulate training. ',
                    'Quantum states are generated using random probability distributions.'
                ], style={'marginBottom': '25px'}),
                
                html.H3('Interactive Features', style={'color': COLORS['sales'], 'marginBottom': '15px'}),
                html.P([
                    'All visualizations update in real-time as parameters are adjusted. ',
                    'Hover over graphs for detailed information. ',
                    'Use the sliders and dropdowns to explore different scenarios.'
                ], style={'marginBottom': '25px'})
            ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'})
        ])
    
    elif tab == 'sales':
        return html.Div([
            color_script,
            # Sidebar with parameters and explanation
            html.Div([
                html.Div([
                    html.H3('Sales Analysis Parameters', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'Adjust these parameters to simulate different sales scenarios. ',
                        'Mean sales affects the central tendency, while standard deviation controls the spread of sales values.'
                    ], style={'marginBottom': '20px'})
                ], style={'marginBottom': '20px'}),
                
                html.Label('Mean Sales', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '12px',
                        'display': 'block'
                    }
                ),
                html.Div(
                    children=[
                        dcc.Slider(
                            id='mean',
                            min=500,
                            max=2000,
                            value=1000,
                            step=100,
                            marks={i: str(i) for i in range(500, 2001, 500)}
                        )
                    ],
                    style=SLIDER_CONTAINER_STYLE
                ),
                
                html.Label('Standard Deviation', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '12px',
                        'display': 'block'
                    }
                ),
                html.Div(
                    children=[
                        dcc.Slider(
                            id='std',
                            min=50,
                            max=500,
                            value=200,
                            step=50,
                            marks={i: str(i) for i in range(50, 501, 150)}
                        )
                    ],
                    style=SLIDER_CONTAINER_STYLE
                ),
            ], style=SIDEBAR_STYLE),
            
            # Main content with graphs and explanations
            html.Div([
                html.Div([
                    html.H3('Sales Trend Analysis', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The line graph shows daily sales over a 3-month period. ',
                        'Higher mean values shift the trend line upward, while higher standard deviation creates more variability in daily sales.'
                    ], style=TEXT_CONTAINER_STYLE)
                ], style=GRAPH_CONTAINER_STYLE),
                dcc.Graph(id='sales-graph'),
                
                html.Div([
                    html.H3('Sales Distribution', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The histogram displays the frequency of different sales amounts. ',
                        'A higher standard deviation results in a wider spread of sales values, ',
                        'while the mean determines the center of the distribution.'
                    ], style=TEXT_CONTAINER_STYLE)
                ], style=GRAPH_CONTAINER_STYLE),
                dcc.Graph(id='dist-graph'),
                
                html.Div([
                    html.H3('Key Statistics', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The statistics below summarize the overall sales performance. ',
                        'Mean sales indicates the average daily revenue, while total sales shows the cumulative performance.'
                    ], style=TEXT_CONTAINER_STYLE)
                ], style=GRAPH_CONTAINER_STYLE),
                html.Div(id='stats', style=STATS_STYLE)
            ], style=MAIN_CONTENT_STYLE)
        ])
    
    elif tab == 'ml':
        return html.Div([
            color_script,
            # Sidebar with parameters and explanation
            html.Div([
                html.Div([
                    html.H3('Machine Learning Configuration', 
                        style={
                            'color': COLORS[tab], 
                            'marginTop': '20px',  # Added top margin to align with main content
                            'marginBottom': '15px'
                        }
                    ),
                    html.P([
                        'Select different ML algorithms and adjust training parameters. ',
                        'Learning rate affects training speed and convergence, while epochs determine training duration.'
                    ], style={'marginBottom': '20px'})
                ], style={'marginBottom': '20px'}),
                
                html.Label('Select Algorithm',
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '12px',
                        'display': 'block'
                    }
                ),
                html.Div(style=DROPDOWN_CONTAINER_STYLE,
                    children=[
                        dcc.Dropdown(
                            id='ml-algorithm',
                            options=[{'label': alg, 'value': alg} for alg in ML_ALGORITHMS],
                            value=ML_ALGORITHMS[0],
                            clearable=False,
                            searchable=False,
                            style=DROPDOWN_STYLE
                        )
                    ]
                ),
                
                html.Label('Learning Rate', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '12px',
                        'display': 'block'
                    }
                ),
                html.Div(
                    children=[
                        dcc.Slider(
                            id='learning-rate',
                            min=0.001,
                            max=0.1,
                            value=0.01,
                            step=0.001,
                            marks={i/100: f'{i/100:.3f}' for i in range(0, 11, 2)}
                        )
                    ],
                    style=SLIDER_CONTAINER_STYLE
                ),
                
                html.Label('Number of Epochs', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '12px',
                        'display': 'block'
                    }
                ),
                html.Div(
                    children=[
                        dcc.Slider(
                            id='epochs',
                            min=10,
                            max=100,
                            value=50,
                            step=10,
                            marks={i: str(i) for i in range(10, 101, 20)}
                        )
                    ],
                    style=SLIDER_CONTAINER_STYLE
                ),
            ], style=SIDEBAR_STYLE),
            
            # Main content with graphs and explanations
            html.Div([
                html.Div([
                    html.H3('Training Accuracy', 
                        style={
                            'color': COLORS[tab], 
                            'marginTop': '20px',  # Added top margin to align with sidebar
                            'marginBottom': '15px'
                        }
                    ),
                    html.P([
                        'This graph shows how the model\'s accuracy improves during training. ',
                        'A higher learning rate leads to faster initial improvement, but may cause instability. ',
                        'More epochs allow the model to learn more complex patterns.'
                    ], style=TEXT_CONTAINER_STYLE)
                ], style=GRAPH_CONTAINER_STYLE),
                dcc.Graph(id='ml-accuracy'),
                
                html.Div([
                    html.H3('Training Loss', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The loss curve indicates how well the model minimizes prediction errors. ',
                        'A decreasing loss shows successful learning, while fluctuations may indicate overfitting.'
                    ], style=TEXT_CONTAINER_STYLE)
                ], style=GRAPH_CONTAINER_STYLE),
                dcc.Graph(id='ml-loss'),
                
                html.Div([
                    html.H3('Model Performance', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The statistics below show the final performance metrics. ',
                        'Higher accuracy indicates better prediction capability, while lower loss suggests better model fit.'
                    ], style=TEXT_CONTAINER_STYLE)
                ], style=GRAPH_CONTAINER_STYLE),
                html.Div(id='ml-stats', style=STATS_STYLE)
            ], style=MAIN_CONTENT_STYLE)
        ])
    
    elif tab == 'dl':
        return html.Div([
            color_script,
            # Sidebar with parameters and explanation
            html.Div([
                html.Div([
                    html.H3('Deep Learning Architecture', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'Explore various deep learning architectures. ',
                        'Batch size affects training stability and memory usage, while network depth controls model complexity.'
                    ], style={'marginBottom': '20px'})
                ], style={'marginBottom': '20px'}),
                
                html.Label('Select Architecture',
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                html.Div(style=DROPDOWN_CONTAINER_STYLE,
                    children=[
                        dcc.Dropdown(
                            id='dl-algorithm',
                            options=[{'label': alg, 'value': alg} for alg in DL_ALGORITHMS],
                            value=DL_ALGORITHMS[0],
                            clearable=False,
                            searchable=False,
                            style=DROPDOWN_STYLE
                        )
                    ]
                ),
                
                html.Label('Batch Size', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                html.Div(style={'margin': '20px 0'},
                    children=[
                        dcc.Slider(
                            id='batch-size',
                            min=16,
                            max=128,
                            value=32,
                            step=16,
                            marks={i: str(i) for i in range(16, 129, 16)}
                        )
                    ]
                ),
                
                html.Label('Network Depth', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                html.Div(style={'margin': '20px 0'},
                    children=[
                        dcc.Slider(
                            id='depth',
                            min=2,
                            max=10,
                            value=5,
                            step=1,
                            marks={i: str(i) for i in range(2, 11, 2)}
                        )
                    ]
                ),
            ], style=SIDEBAR_STYLE),
            
            # Main content with graphs and explanations
            html.Div([
                html.Div([
                    html.H3('Training Progress', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'This graph shows the model\'s learning progress on the training data. ',
                        'Larger batch sizes may lead to smoother learning curves, ',
                        'while deeper networks can capture more complex patterns.'
                    ], style={'marginBottom': '20px'})
                ]),
                dcc.Graph(id='dl-training'),
                
                html.Div([
                    html.H3('Validation Performance', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The validation curve indicates how well the model generalizes to new data. ',
                        'A gap between training and validation performance may indicate overfitting.'
                    ], style={'marginBottom': '20px'})
                ]),
                dcc.Graph(id='dl-validation'),
                
                html.Div([
                    html.H3('Architecture Performance', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The statistics below compare training and validation performance. ',
                        'Good generalization is indicated by similar training and validation accuracy.'
                    ], style={'marginBottom': '20px'})
                ]),
                html.Div(id='dl-stats', style=STATS_STYLE)
            ], style=MAIN_CONTENT_STYLE)
        ])
    
    elif tab == 'qc':
        return html.Div([
            color_script,
            # Sidebar with parameters and explanation
            html.Div([
                html.Div([
                    html.H3('Quantum Circuit Parameters', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'Configure quantum circuit parameters. ',
                        'Number of qubits determines system size, while circuit depth affects computation complexity.'
                    ], style={'marginBottom': '20px'})
                ], style={'marginBottom': '20px'}),
                
                html.Label('Select Algorithm',
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                html.Div(style=DROPDOWN_CONTAINER_STYLE,
                    children=[
                        dcc.Dropdown(
                            id='qc-algorithm',
                            options=[{'label': alg, 'value': alg} for alg in QC_ALGORITHMS],
                            value=QC_ALGORITHMS[0],
                            clearable=False,
                            searchable=False,
                            style=DROPDOWN_STYLE
                        )
                    ]
                ),
                
                html.Label('Number of Qubits', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                html.Div(style={'margin': '20px 0'},
                    children=[
                        dcc.Slider(
                            id='qubits',
                            min=2,
                            max=10,
                            value=4,
                            step=1,
                            marks={i: str(i) for i in range(2, 11, 2)}
                        )
                    ]
                ),
                
                html.Label('Circuit Depth', 
                    style={
                        'color': COLORS[tab],
                        'fontSize': '1.1em',
                        'fontWeight': '500',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                html.Div(style={'margin': '20px 0'},
                    children=[
                        dcc.Slider(
                            id='circuit-depth',
                            min=1,
                            max=10,
                            value=3,
                            step=1,
                            marks={i: str(i) for i in range(1, 11, 2)}
                        )
                    ]
                ),
            ], style=SIDEBAR_STYLE),
            
            # Main content with graphs and explanations
            html.Div([
                html.Div([
                    html.H3('Quantum State Probabilities', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'This visualization shows the probability distribution of quantum states. ',
                        'Each bar represents the probability of measuring a specific quantum state. ',
                        'More qubits create a larger state space with more possible outcomes.'
                    ], style={'marginBottom': '20px'})
                ]),
                dcc.Graph(id='qc-circuit'),
                
                html.Div([
                    html.H3('State Distribution', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The scatter plot provides an alternative view of quantum state probabilities. ',
                        'Higher circuit depth can create more complex quantum states, ',
                        'while more qubits increase the dimensionality of the quantum system.'
                    ], style={'marginBottom': '20px'})
                ]),
                dcc.Graph(id='qc-state'),
                
                html.Div([
                    html.H3('Quantum Metrics', style={'color': COLORS[tab], 'marginBottom': '15px'}),
                    html.P([
                        'The statistics below show key quantum metrics. ',
                        'Fidelity measures how close the quantum state is to the target state, ',
                        'while entropy indicates the amount of quantum information.'
                    ], style={'marginBottom': '20px'})
                ]),
                html.Div(id='qc-stats', style=STATS_STYLE)
            ], style=MAIN_CONTENT_STYLE)
        ])

@app.callback(
    [Output('sales-graph', 'figure'),
     Output('dist-graph', 'figure'),
     Output('stats', 'children')],
    [Input('mean', 'value'),
     Input('std', 'value')]
)
def update_sales(mean, std):
    dates = pd.date_range('2024-01-01', '2024-03-31')
    sales = np.random.normal(1000 if mean is None else mean, 
                           200 if std is None else std, 
                           len(dates))
    sales = np.maximum(sales, 0)
    df = pd.DataFrame({'Date': dates, 'Sales': sales})

    fig1 = {
        'data': [
            {
                'x': df['Date'],
                'y': df['Sales'],
                'type': 'scatter',
                'mode': 'lines',
                'line': {'color': COLORS['sales'], 'width': 2}
            }
        ],
        'layout': {
            'title': 'Sales Trend',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
            'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40}
        }
    }

    fig2 = {
        'data': [
            {
                'x': df['Sales'],
                'type': 'histogram',
                'marker': {
                    'color': COLORS['sales'],
                    'line': {
                        'color': COLORS['accent1'],
                        'width': 1
                    }
                },
                'opacity': 0.8,
                'nbinsx': 20
            }
        ],
        'layout': {
            'title': 'Sales Distribution',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
            'bargap': 0.1,
            'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40}
        }
    }

    stats_div = html.Div([
        html.H3(f'Mean Sales: ${df["Sales"].mean():,.2f}', 
            style=get_stats_style(COLORS['sales'])),
        html.H3(f'Total Sales: ${df["Sales"].sum():,.2f}', 
            style=get_stats_style(COLORS['accent1']))
    ])

    return fig1, fig2, stats_div

@app.callback(
    [Output('ml-accuracy', 'figure'),
     Output('ml-loss', 'figure'),
     Output('ml-stats', 'children')],
    [Input('ml-algorithm', 'value'),
     Input('learning-rate', 'value'),
     Input('epochs', 'value')]
)
def update_ml(algorithm, lr, epochs):
    epochs = 50 if epochs is None else epochs
    lr = 0.01 if lr is None else lr
    algorithm = ML_ALGORITHMS[0] if algorithm is None else algorithm
    
    x = np.linspace(0, epochs, epochs)
    accuracy = 1 - 0.3 * np.exp(-0.5 * lr * x) + np.random.normal(0, 0.01, epochs)
    loss = 0.3 * np.exp(-0.5 * lr * x) + np.random.normal(0, 0.01, epochs)

    fig1 = {
        'data': [
            {
                'x': x,
                'y': accuracy,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Accuracy',
                'line': {'color': COLORS['sales']}
            }
        ],
        'layout': {
            'title': f'{algorithm} - Training Accuracy',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
        }
    }

    fig2 = {
        'data': [
            {
                'x': x,
                'y': loss,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Loss',
                'line': {'color': COLORS['accent2']}
            }
        ],
        'layout': {
            'title': f'{algorithm} - Training Loss',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
        }
    }

    stats_div = html.Div([
        html.H3(f'Algorithm: {algorithm}', style={'color': COLORS['sales']}),
        html.H3(f'Final Accuracy: {accuracy[-1]:.2%}', style={'color': COLORS['sales']}),
        html.H3(f'Final Loss: {loss[-1]:.4f}', style={'color': COLORS['accent2']})
    ])

    return fig1, fig2, stats_div

@app.callback(
    [Output('dl-training', 'figure'),
     Output('dl-validation', 'figure'),
     Output('dl-stats', 'children')],
    [Input('dl-algorithm', 'value'),
     Input('batch-size', 'value'),
     Input('depth', 'value')]
)
def update_dl(algorithm, batch_size, depth):
    algorithm = DL_ALGORITHMS[0] if algorithm is None else algorithm
    batch_size = 32 if batch_size is None else batch_size
    depth = 5 if depth is None else depth
    
    epochs = 50
    x = np.linspace(0, epochs, epochs)
    train_acc = 1 - 0.4 * np.exp(-0.15 * x) + np.random.normal(0, 0.02, epochs)
    val_acc = train_acc - 0.1 + np.random.normal(0, 0.02, epochs)

    fig1 = {
        'data': [
            {
                'x': x,
                'y': train_acc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Training',
                'line': {'color': COLORS['sales']}
            }
        ],
        'layout': {
            'title': f'{algorithm} - Training Accuracy',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
        }
    }

    fig2 = {
        'data': [
            {
                'x': x,
                'y': val_acc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Validation',
                'line': {'color': COLORS['accent2']}
            }
        ],
        'layout': {
            'title': f'{algorithm} - Validation Accuracy',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
        }
    }

    stats_div = html.Div([
        html.H3(f'Architecture: {algorithm}', style={'color': COLORS['sales']}),
        html.H3(f'Training Accuracy: {train_acc[-1]:.2%}', style={'color': COLORS['sales']}),
        html.H3(f'Validation Accuracy: {val_acc[-1]:.2%}', style={'color': COLORS['accent2']})
    ])

    return fig1, fig2, stats_div

@app.callback(
    [Output('qc-circuit', 'figure'),
     Output('qc-state', 'figure'),
     Output('qc-stats', 'children')],
    [Input('qc-algorithm', 'value'),
     Input('qubits', 'value'),
     Input('circuit-depth', 'value')]
)
def update_qc(algorithm, qubits, depth):
    algorithm = QC_ALGORITHMS[0] if algorithm is None else algorithm
    qubits = 4 if qubits is None else qubits
    depth = 3 if depth is None else depth
    
    states = 2 ** qubits
    probabilities = np.random.dirichlet(np.ones(states))
    state_labels = [f'|{format(i, "0"+str(qubits)+"b")}⟩' for i in range(states)]

    fig1 = {
        'data': [
            {
                'x': state_labels,
                'y': probabilities,
                'type': 'bar',
                'marker': {'color': COLORS['sales']}
            }
        ],
        'layout': {
            'title': f'{algorithm} - Quantum State Probabilities',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
        }
    }

    fig2 = {
        'data': [
            {
                'x': state_labels,
                'y': probabilities,
                'type': 'scatter',
                'mode': 'markers',
                'marker': {
                    'color': COLORS['accent2'],
                    'size': 10
                }
            }
        ],
        'layout': {
            'title': f'{algorithm} - State Distribution',
            'paper_bgcolor': COLORS['background'],
            'plot_bgcolor': COLORS['secondary_background'],
            'font': {'color': COLORS['sales']},
        }
    }

    fidelity = np.sum(probabilities ** 2)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    stats_div = html.Div([
        html.H3(f'Algorithm: {algorithm}', style={'color': COLORS['sales']}),
        html.H3(f'Quantum Fidelity: {fidelity:.4f}', style={'color': COLORS['sales']}),
        html.H3(f'Von Neumann Entropy: {entropy:.4f}', style={'color': COLORS['accent2']})
    ])

    return fig1, fig2, stats_div

# Update stats style with new typography
def get_stats_style(color):
    return {
        'color': color,
        'fontSize': '1.15em',
        'fontWeight': '500',
        'fontFamily': 'Inter, sans-serif',
        'marginBottom': '10px',
        'letterSpacing': '0.2px'
    }

if __name__ == '__main__':
    app.run_server(debug=True) 