import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid, Paper, Typography, Select, MenuItem, FormControl, InputLabel, Button, Slider, TextField } from '@material-ui/core';
import Plot from 'react-plotly.js';
import axios from 'axios';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(3),
    background: '#ffffff',
    minHeight: '100vh',
  },
  paper: {
    padding: theme.spacing(3),
    margin: theme.spacing(2),
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  title: {
    marginBottom: theme.spacing(2),
    color: '#1a237e',
    fontWeight: 600,
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 200,
  },
}));

function MLPredictions() {
  const classes = useStyles();
  const [selectedModel, setSelectedModel] = useState('linear_regression');
  const [parameters, setParameters] = useState({
    learningRate: 0.01,
    epochs: 100,
    batchSize: 32,
    regularization: 0.1,
  });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    // Generate initial data
    const initialData = Array.from({ length: 100 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      label: Math.random() > 0.5 ? 0 : 1,
    }));
    setData(initialData);
  }, []);

  const handleParameterChange = (param) => (event, newValue) => {
    setParameters({
      ...parameters,
      [param]: newValue,
    });
  };

  const handlePredict = () => {
    // Generate random performance metrics
    const simulatedPrediction = {
      accuracy: Math.random() * 0.5 + 0.5,
      precision: Math.random() * 0.5 + 0.5,
      recall: Math.random() * 0.5 + 0.5,
      f1: Math.random() * 0.5 + 0.5,
      confusionMatrix: [
        [Math.floor(Math.random() * 100), Math.floor(Math.random() * 50)],
        [Math.floor(Math.random() * 50), Math.floor(Math.random() * 100)],
      ],
    };
    setPrediction(simulatedPrediction);
  };

  return (
    <div className={classes.root}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h5" className={classes.title}>
              Machine Learning Predictions
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl className={classes.formControl}>
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <MenuItem value="linear_regression">Linear Regression</MenuItem>
                    <MenuItem value="logistic_regression">Logistic Regression</MenuItem>
                    <MenuItem value="random_forest">Random Forest</MenuItem>
                    <MenuItem value="svm">Support Vector Machine</MenuItem>
                    <MenuItem value="neural_network">Neural Network</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handlePredict}
                  style={{ marginTop: 16 }}
                >
                  Run Prediction
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {data && (
          <Grid item xs={12} md={6}>
            <Paper className={classes.paper}>
              <Typography variant="h6" className={classes.title}>
                Data Distribution
              </Typography>
              <div style={{ height: 400 }}>
                <Plot
                  data={[
                    {
                      x: data.map((d) => d.x),
                      y: data.map((d) => d.y),
                      type: 'scatter',
                      mode: 'markers',
                      name: 'Data Points',
                      marker: { 
                        size: 8,
                        color: data.map(d => d.label === 0 ? '#3498db' : '#e74c3c')
                      }
                    }
                  ]}
                  layout={{
                    title: 'Data Distribution',
                    xaxis: { title: 'Feature 1' },
                    yaxis: { title: 'Feature 2' },
                    showlegend: true
                  }}
                />
              </div>
            </Paper>
          </Grid>
        )}

        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Model Performance
            </Typography>
            <div style={{ height: 400 }}>
              <Plot
                data={[
                  {
                    x: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    y: prediction ? [
                      prediction.accuracy,
                      prediction.precision,
                      prediction.recall,
                      prediction.f1
                    ] : [0.85, 0.82, 0.88, 0.85],
                    type: 'bar',
                    name: 'Metrics',
                    marker: {
                      color: 'rgba(52, 152, 219, 0.7)',
                      line: {
                        color: 'rgba(52, 152, 219, 1)',
                        width: 1
                      }
                    }
                  }
                ]}
                layout={{
                  title: 'Model Performance Metrics',
                  yaxis: { title: 'Score', range: [0, 1] }
                }}
              />
            </div>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Confusion Matrix
            </Typography>
            <div style={{ height: 400 }}>
              <Plot
                data={[
                  {
                    z: prediction ? prediction.confusionMatrix : [[50, 10], [5, 35]],
                    x: ['Predicted Negative', 'Predicted Positive'],
                    y: ['Actual Negative', 'Actual Positive'],
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    showscale: true
                  }
                ]}
                layout={{
                  title: 'Confusion Matrix',
                  xaxis: { title: 'Predicted' },
                  yaxis: { title: 'Actual' }
                }}
              />
            </div>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Learning Curve
            </Typography>
            <div style={{ height: 400 }}>
              <Plot
                data={[
                  {
                    x: Array.from({ length: 10 }, (_, i) => i * 10),
                    y: Array.from({ length: 10 }, () => Math.random() * 0.2 + 0.8),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Training',
                    line: { color: '#3498db' }
                  },
                  {
                    x: Array.from({ length: 10 }, (_, i) => i * 10),
                    y: Array.from({ length: 10 }, () => Math.random() * 0.2 + 0.7),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Validation',
                    line: { color: '#e74c3c' }
                  }
                ]}
                layout={{
                  title: 'Learning Curve',
                  xaxis: { title: 'Training Size' },
                  yaxis: { title: 'Score', range: [0, 1] }
                }}
              />
            </div>
          </Paper>
        </Grid>
      </Grid>
    </div>
  );
}

export default MLPredictions; 