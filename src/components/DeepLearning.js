import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid, Paper, Typography, Select, MenuItem, FormControl, InputLabel, Button, Slider } from '@material-ui/core';
import Plot from 'react-plotly.js';
import axios from 'axios';

const useStyles = makeStyles((theme) => ({
  paper: {
    padding: theme.spacing(3),
    margin: theme.spacing(1),
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 200,
  },
  title: {
    marginBottom: theme.spacing(2),
    color: '#1a237e',
    fontWeight: 600,
  },
  subtitle: {
    color: '#666',
    marginBottom: theme.spacing(2),
  },
  slider: {
    width: '100%',
    marginTop: theme.spacing(2),
  },
  button: {
    marginTop: theme.spacing(2),
  },
}));

function DeepLearning() {
  const classes = useStyles();
  const [selectedModel, setSelectedModel] = useState('cnn');
  const [parameters, setParameters] = useState({
    learningRate: 0.001,
    epochs: 50,
    batchSize: 32,
    dropout: 0.2,
    layers: 3,
  });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [trainingHistory, setTrainingHistory] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch sample data from a public API
      const response = await axios.get('https://api.sampleapis.com/coffee/hot');
      const dlData = response.data.map(item => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        label: Math.random() > 0.5 ? 0 : 1,
      }));
      setData(dlData);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
    setLoading(false);
  };

  const handleParameterChange = (param) => (event, newValue) => {
    setParameters({
      ...parameters,
      [param]: newValue,
    });
  };

  const handleTrain = () => {
    // Simulate training history
    const history = {
      loss: Array.from({ length: parameters.epochs }, () => Math.random() * 0.5),
      accuracy: Array.from({ length: parameters.epochs }, () => Math.random() * 0.5 + 0.5),
      val_loss: Array.from({ length: parameters.epochs }, () => Math.random() * 0.5),
      val_accuracy: Array.from({ length: parameters.epochs }, () => Math.random() * 0.5 + 0.5),
    };
    setTrainingHistory(history);
  };

  const renderVisualizations = () => {
    if (!data) return null;

    return (
      <Grid container spacing={3}>
        {/* Model Architecture */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Model Architecture
            </Typography>
            <Plot
              data={[
                {
                  x: ['Input', 'Conv1', 'Conv2', 'Conv3', 'Dense', 'Output'],
                  y: [32, 64, 128, 64, 32, 10],
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Layer Size',
                  line: { color: '#2196f3' },
                },
              ]}
              layout={{
                title: 'Model Architecture',
                xaxis: { title: 'Layer' },
                yaxis: { title: 'Units/Channels' },
              }}
            />
          </Paper>
        </Grid>

        {/* Training Progress */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Training Progress
            </Typography>
            {trainingHistory && (
              <Plot
                data={[
                  {
                    x: Array.from({ length: parameters.epochs }, (_, i) => i),
                    y: trainingHistory.loss,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Training Loss',
                    line: { color: '#f44336' },
                  },
                  {
                    x: Array.from({ length: parameters.epochs }, (_, i) => i),
                    y: trainingHistory.val_loss,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Validation Loss',
                    line: { color: '#ff9800' },
                  },
                ]}
                layout={{
                  title: 'Loss Over Time',
                  xaxis: { title: 'Epoch' },
                  yaxis: { title: 'Loss' },
                }}
              />
            )}
          </Paper>
        </Grid>

        {/* Accuracy Metrics */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Accuracy Metrics
            </Typography>
            {trainingHistory && (
              <Plot
                data={[
                  {
                    x: Array.from({ length: parameters.epochs }, (_, i) => i),
                    y: trainingHistory.accuracy,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Training Accuracy',
                    line: { color: '#4caf50' },
                  },
                  {
                    x: Array.from({ length: parameters.epochs }, (_, i) => i),
                    y: trainingHistory.val_accuracy,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Validation Accuracy',
                    line: { color: '#8bc34a' },
                  },
                ]}
                layout={{
                  title: 'Accuracy Over Time',
                  xaxis: { title: 'Epoch' },
                  yaxis: { title: 'Accuracy', range: [0, 1] },
                }}
              />
            )}
          </Paper>
        </Grid>

        {/* Feature Maps */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Feature Maps
            </Typography>
            <Plot
              data={[
                {
                  z: Array.from({ length: 8 }, () =>
                    Array.from({ length: 8 }, () => Math.random())
                  ),
                  type: 'heatmap',
                  colorscale: 'Viridis',
                },
              ]}
              layout={{
                title: 'Convolutional Feature Maps',
              }}
            />
          </Paper>
        </Grid>
      </Grid>
    );
  };

  return (
    <div className={classes.root}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h5" gutterBottom>
              Deep Learning Models
            </Typography>
            <Typography variant="body1" paragraph>
              This dashboard demonstrates various deep learning architectures and their applications.
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h6" gutterBottom>Model Architecture</Typography>
            <Typography variant="body2" paragraph>
              Choose from different deep learning architectures, each designed for specific types of data and tasks:
            </Typography>
            <Typography variant="body2" component="div" paragraph>
              <ul>
                <li><strong>CNN (Convolutional Neural Network):</strong> Specialized for image and spatial data processing. Uses convolutional layers to detect features hierarchically.</li>
                <li><strong>RNN (Recurrent Neural Network):</strong> Designed for sequential data like time series. Maintains memory of previous inputs through recurrent connections.</li>
                <li><strong>LSTM (Long Short-Term Memory):</strong> Advanced RNN with memory cells that can learn long-term dependencies. Prevents vanishing gradient problem.</li>
                <li><strong>Transformer:</strong> Uses self-attention mechanism to process all data points simultaneously. Excellent for complex pattern recognition.</li>
                <li><strong>GAN (Generative Adversarial Network):</strong> Two networks (generator and discriminator) that compete to create realistic synthetic data.</li>
              </ul>
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Model Type</InputLabel>
                  <Select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                    <MenuItem value="cnn">CNN</MenuItem>
                    <MenuItem value="rnn">RNN</MenuItem>
                    <MenuItem value="lstm">LSTM</MenuItem>
                    <MenuItem value="transformer">Transformer</MenuItem>
                    <MenuItem value="gan">GAN</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h6" gutterBottom>Model Parameters</Typography>
            <Typography variant="body2" paragraph>
              Adjust these parameters to optimize your model's performance:
            </Typography>
            <Typography variant="body2" component="div" paragraph>
              <ul>
                <li><strong>Learning Rate:</strong> Controls the step size in gradient descent. Lower values are more stable but slower to converge.</li>
                <li><strong>Epochs:</strong> Number of complete passes through the training dataset. More epochs can improve accuracy but may lead to overfitting.</li>
                <li><strong>Batch Size:</strong> Number of samples processed before updating the model. Affects training stability and memory usage.</li>
                <li><strong>Dropout Rate:</strong> Percentage of neurons randomly deactivated during training. Helps prevent overfitting.</li>
                <li><strong>Number of Layers:</strong> Depth of the neural network. More layers can learn more complex patterns but require more data and computation.</li>
              </ul>
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Learning Rate: {parameters.learningRate}</Typography>
                <Slider
                  value={parameters.learningRate}
                  onChange={handleParameterChange('learningRate')}
                  min={0.0001}
                  max={0.1}
                  step={0.0001}
                  marks={[
                    { value: 0.0001, label: '0.0001' },
                    { value: 0.1, label: '0.1' }
                  ]}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Epochs: {parameters.epochs}</Typography>
                <Slider
                  value={parameters.epochs}
                  onChange={handleParameterChange('epochs')}
                  min={10}
                  max={1000}
                  step={10}
                  marks={[
                    { value: 10, label: '10' },
                    { value: 1000, label: '1000' }
                  ]}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Batch Size: {parameters.batchSize}</Typography>
                <Slider
                  value={parameters.batchSize}
                  onChange={handleParameterChange('batchSize')}
                  min={8}
                  max={128}
                  step={8}
                  marks={[
                    { value: 8, label: '8' },
                    { value: 128, label: '128' }
                  ]}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Dropout Rate: {parameters.dropout}</Typography>
                <Slider
                  value={parameters.dropout}
                  onChange={handleParameterChange('dropout')}
                  min={0}
                  max={0.5}
                  step={0.1}
                  marks={[
                    { value: 0, label: '0' },
                    { value: 0.5, label: '0.5' }
                  ]}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>Number of Layers: {parameters.layers}</Typography>
                <Slider
                  value={parameters.layers}
                  onChange={handleParameterChange('layers')}
                  min={1}
                  max={10}
                  step={1}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 10, label: '10' }
                  ]}
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" gutterBottom>Model Architecture</Typography>
            <Typography variant="body2" paragraph>
              The diagram below shows the structure of your selected model, including:
            </Typography>
            <Typography variant="body2" component="div" paragraph>
              <ul>
                <li>Input and output layers</li>
                <li>Hidden layers with their sizes</li>
                <li>Activation functions</li>
                <li>Connection patterns</li>
              </ul>
            </Typography>
            <div style={{ height: 400 }}>
              <Plot
                data={[
                  {
                    x: [1, 2, 3, 4, 5],
                    y: [1, 2, 3, 2, 1],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Architecture',
                    line: { shape: 'spline' }
                  }
                ]}
                layout={{
                  title: 'Model Architecture',
                  xaxis: { title: 'Layer' },
                  yaxis: { title: 'Neurons' }
                }}
              />
            </div>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" gutterBottom>Training Progress</Typography>
            <Typography variant="body2" paragraph>
              Monitor your model's training progress through these metrics:
            </Typography>
            <Typography variant="body2" component="div" paragraph>
              <ul>
                <li><strong>Loss:</strong> Measures how well the model fits the training data</li>
                <li><strong>Accuracy:</strong> Percentage of correct predictions</li>
                <li><strong>Validation Metrics:</strong> Performance on unseen data</li>
              </ul>
            </Typography>
            <div style={{ height: 400 }}>
              <Plot
                data={[
                  {
                    x: Array.from({ length: 10 }, (_, i) => i + 1),
                    y: Array.from({ length: 10 }, () => Math.random() * 0.5 + 0.5),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Loss',
                    line: { shape: 'spline' }
                  },
                  {
                    x: Array.from({ length: 10 }, (_, i) => i + 1),
                    y: Array.from({ length: 10 }, () => Math.random() * 0.2 + 0.8),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Accuracy',
                    line: { shape: 'spline' }
                  }
                ]}
                layout={{
                  title: 'Training Progress',
                  xaxis: { title: 'Epoch' },
                  yaxis: { title: 'Value', range: [0, 1] },
                  showlegend: true
                }}
              />
            </div>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h6" gutterBottom>Feature Maps</Typography>
            <Typography variant="body2" paragraph>
              Feature maps show how the model processes and transforms input data through its layers:
            </Typography>
            <Typography variant="body2" component="div" paragraph>
              <ul>
                <li>Each layer extracts different features from the input</li>
                <li>Early layers detect simple patterns (edges, colors)</li>
                <li>Later layers combine these into complex features</li>
                <li>Visualization helps understand what the model learns</li>
              </ul>
            </Typography>
            <div style={{ height: 400 }}>
              <Plot
                data={[
                  {
                    z: Array.from({ length: 8 }, () => 
                      Array.from({ length: 8 }, () => Math.random())
                    ),
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    showscale: true
                  }
                ]}
                layout={{
                  title: 'Feature Maps',
                  xaxis: { title: 'Width' },
                  yaxis: { title: 'Height' }
                }}
              />
            </div>
          </Paper>
        </Grid>
      </Grid>
    </div>
  );
}

export default DeepLearning; 