import React, { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid, Paper, Typography, Select, MenuItem, FormControl, InputLabel, Button, Slider } from '@material-ui/core';
import Plot from 'react-plotly.js';

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
  circuit: {
    margin: theme.spacing(2),
    padding: theme.spacing(2),
    backgroundColor: '#f5f5f5',
    borderRadius: theme.spacing(1),
    fontFamily: 'monospace',
  },
}));

function QuantumComputing() {
  const classes = useStyles();
  const [selectedCircuit, setSelectedCircuit] = useState('superposition');
  const [parameters, setParameters] = useState({
    qubits: 2,
    depth: 3,
    shots: 1000,
    noise: 0.1,
  });
  const [results, setResults] = useState(null);

  const handleParameterChange = (param) => (event, newValue) => {
    setParameters({
      ...parameters,
      [param]: newValue,
    });
  };

  const runCircuit = () => {
    // Simulate quantum circuit results
    let simulatedResults;
    switch (selectedCircuit) {
      case 'superposition':
        simulatedResults = {
          counts: { '0': 50, '1': 50 },
          statevector: [0.707, 0.707],
          probabilities: [0.5, 0.5],
        };
        break;
      case 'entanglement':
        simulatedResults = {
          counts: { '00': 50, '11': 50 },
          statevector: [0.707, 0, 0, 0.707],
          probabilities: [0.5, 0, 0, 0.5],
        };
        break;
      case 'quantum_fourier':
        simulatedResults = {
          counts: { '00': 25, '01': 25, '10': 25, '11': 25 },
          statevector: [0.5, 0.5, 0.5, 0.5],
          probabilities: [0.25, 0.25, 0.25, 0.25],
        };
        break;
      case 'grover':
        simulatedResults = {
          counts: { '00': 10, '01': 10, '10': 70, '11': 10 },
          statevector: [0.316, 0.316, 0.837, 0.316],
          probabilities: [0.1, 0.1, 0.7, 0.1],
        };
        break;
      case 'shor':
        simulatedResults = {
          counts: { '000': 20, '001': 20, '010': 20, '011': 20, '100': 20 },
          statevector: [0.447, 0.447, 0.447, 0.447, 0.447],
          probabilities: [0.2, 0.2, 0.2, 0.2, 0.2],
        };
        break;
      default:
        simulatedResults = {};
    }
    setResults(simulatedResults);
  };

  const renderVisualizations = () => {
    if (!results) return null;

    return (
      <Grid container spacing={3}>
        {/* Measurement Results */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Measurement Results
            </Typography>
            <Plot
              data={[
                {
                  x: Object.keys(results.counts),
                  y: Object.values(results.counts),
                  type: 'bar',
                  name: 'Counts',
                  marker: { color: '#2196f3' },
                },
              ]}
              layout={{
                title: 'Measurement Results',
                xaxis: { title: 'State' },
                yaxis: { title: 'Counts' },
              }}
            />
          </Paper>
        </Grid>

        {/* State Vector */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              State Vector
            </Typography>
            <Plot
              data={[
                {
                  x: Array.from({ length: results.statevector.length }, (_, i) => i),
                  y: results.statevector,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Amplitude',
                  line: { color: '#4caf50' },
                },
              ]}
              layout={{
                title: 'State Vector Amplitude',
                xaxis: { title: 'State' },
                yaxis: { title: 'Amplitude' },
              }}
            />
          </Paper>
        </Grid>

        {/* Probability Distribution */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Probability Distribution
            </Typography>
            <Plot
              data={[
                {
                  x: Array.from({ length: results.probabilities.length }, (_, i) => i),
                  y: results.probabilities,
                  type: 'bar',
                  name: 'Probability',
                  marker: { color: '#ff9800' },
                },
              ]}
              layout={{
                title: 'State Probabilities',
                xaxis: { title: 'State' },
                yaxis: { title: 'Probability', range: [0, 1] },
              }}
            />
          </Paper>
        </Grid>

        {/* Circuit Diagram */}
        <Grid item xs={12} md={6}>
          <Paper className={classes.paper}>
            <Typography variant="h6" className={classes.title}>
              Circuit Diagram
            </Typography>
            <div className={classes.circuit}>
              <Typography>
                {selectedCircuit === 'superposition' && 'H |0⟩ → (|0⟩ + |1⟩)/√2'}
                {selectedCircuit === 'entanglement' && 'H |0⟩ ⊗ |0⟩ → (|00⟩ + |11⟩)/√2'}
                {selectedCircuit === 'quantum_fourier' && 'QFT |0⟩ → (|0⟩ + |1⟩ + |2⟩ + |3⟩)/2'}
                {selectedCircuit === 'grover' && 'G |ψ⟩ → |ψ⟩ - 2|s⟩⟨s|ψ⟩'}
                {selectedCircuit === 'shor' && 'U |x⟩|0⟩ → |x⟩|f(x)⟩'}
              </Typography>
            </div>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  return (
    <div>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h5" className={classes.title}>
              Quantum Computing
            </Typography>
            <Typography variant="subtitle1" className={classes.subtitle}>
              Simulate and visualize quantum circuits and algorithms
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl className={classes.formControl}>
                  <InputLabel>Select Circuit</InputLabel>
                  <Select
                    value={selectedCircuit}
                    onChange={(e) => setSelectedCircuit(e.target.value)}
                  >
                    <MenuItem value="superposition">Superposition</MenuItem>
                    <MenuItem value="entanglement">Entanglement</MenuItem>
                    <MenuItem value="quantum_fourier">Quantum Fourier</MenuItem>
                    <MenuItem value="grover">Grover's Algorithm</MenuItem>
                    <MenuItem value="shor">Shor's Algorithm</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Number of Qubits</Typography>
                <Slider
                  value={parameters.qubits}
                  onChange={handleParameterChange('qubits')}
                  min={1}
                  max={5}
                  step={1}
                  className={classes.slider}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Circuit Depth</Typography>
                <Slider
                  value={parameters.depth}
                  onChange={handleParameterChange('depth')}
                  min={1}
                  max={10}
                  step={1}
                  className={classes.slider}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Number of Shots</Typography>
                <Slider
                  value={parameters.shots}
                  onChange={handleParameterChange('shots')}
                  min={100}
                  max={10000}
                  step={100}
                  className={classes.slider}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography gutterBottom>Noise Level</Typography>
                <Slider
                  value={parameters.noise}
                  onChange={handleParameterChange('noise')}
                  min={0}
                  max={1}
                  step={0.1}
                  className={classes.slider}
                />
              </Grid>
            </Grid>
            <Button
              variant="contained"
              color="primary"
              onClick={runCircuit}
              className={classes.button}
            >
              Run Circuit
            </Button>
          </Paper>
        </Grid>
      </Grid>

      {renderVisualizations()}
    </div>
  );
}

export default QuantumComputing; 