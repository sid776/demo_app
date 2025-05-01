import React, { useState, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid, Paper, Typography, Select, MenuItem, FormControl, InputLabel, Button } from '@material-ui/core';
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
  description: {
    marginTop: theme.spacing(2),
    color: '#666',
    lineHeight: 1.6,
  },
  graphContainer: {
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: theme.spacing(3),
  },
  graph: {
    flex: 1,
  },
  textContainer: {
    flex: 1,
    padding: theme.spacing(2),
  },
}));

function SalesAnalysis() {
  const classes = useStyles();
  const [selectedMetric, setSelectedMetric] = useState('revenue');
  const [timeFrame, setTimeFrame] = useState('monthly');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Generate sample data immediately
    const sampleData = Array.from({ length: 30 }, (_, i) => ({
      date: new Date(2023, 0, i + 1),
      revenue: Math.floor(Math.random() * 10000) + 1000,
      units: Math.floor(Math.random() * 100) + 10,
      profit: Math.floor(Math.random() * 5000) + 500,
      customers: Math.floor(Math.random() * 50) + 5,
    }));
    setData(sampleData);
  }, []);

  const handleMetricChange = (event) => {
    setSelectedMetric(event.target.value);
  };

  const handleTimeFrameChange = (event) => {
    setTimeFrame(event.target.value);
  };

  return (
    <div className={classes.root}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Typography variant="h5" className={classes.title}>
              Sales Analysis Dashboard
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl className={classes.formControl}>
                  <InputLabel>Metric</InputLabel>
                  <Select
                    value={selectedMetric}
                    onChange={handleMetricChange}
                  >
                    <MenuItem value="revenue">Revenue</MenuItem>
                    <MenuItem value="units">Units Sold</MenuItem>
                    <MenuItem value="profit">Profit</MenuItem>
                    <MenuItem value="customers">Customers</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl className={classes.formControl}>
                  <InputLabel>Time Frame</InputLabel>
                  <Select
                    value={timeFrame}
                    onChange={handleTimeFrameChange}
                  >
                    <MenuItem value="daily">Daily</MenuItem>
                    <MenuItem value="weekly">Weekly</MenuItem>
                    <MenuItem value="monthly">Monthly</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {data && (
          <Grid item xs={12} md={6}>
            <Paper className={classes.paper}>
              <Typography variant="h6" className={classes.title}>
                Sales Trend
              </Typography>
              <div className={classes.graphContainer}>
                <div className={classes.graph} style={{ height: 400 }}>
                  <Plot
                    data={[
                      {
                        x: data.map(d => d.date),
                        y: data.map(d => d[selectedMetric]),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: selectedMetric,
                        line: { color: '#3498db' },
                        marker: { size: 8 }
                      }
                    ]}
                    layout={{
                      title: `${selectedMetric} Over Time`,
                      xaxis: { title: 'Date' },
                      yaxis: { title: selectedMetric }
                    }}
                  />
                </div>
                <div className={classes.textContainer}>
                  <Typography variant="body1" className={classes.description}>
                    This line chart shows the trend of {selectedMetric} over time. The blue line represents the overall trend, while the markers show individual data points. Use this visualization to:
                  </Typography>
                  <ul>
                    <li>Identify patterns and trends in your sales data</li>
                    <li>Spot seasonal variations or anomalies</li>
                    <li>Track growth or decline over time</li>
                    <li>Compare performance across different time periods</li>
                  </ul>
                </div>
              </div>
            </Paper>
          </Grid>
        )}

        {data && (
          <Grid item xs={12} md={6}>
            <Paper className={classes.paper}>
              <Typography variant="h6" className={classes.title}>
                Distribution
              </Typography>
              <div className={classes.graphContainer}>
                <div className={classes.graph} style={{ height: 400 }}>
                  <Plot
                    data={[
                      {
                        x: data.map(d => d[selectedMetric]),
                        type: 'histogram',
                        name: selectedMetric,
                        marker: { color: '#3498db' }
                      }
                    ]}
                    layout={{
                      title: `${selectedMetric} Distribution`,
                      xaxis: { title: selectedMetric },
                      yaxis: { title: 'Frequency' }
                    }}
                  />
                </div>
                <div className={classes.textContainer}>
                  <Typography variant="body1" className={classes.description}>
                    This histogram shows how your {selectedMetric} is distributed across different value ranges. The height of each bar represents how frequently values fall within that range. Use this visualization to:
                  </Typography>
                  <ul>
                    <li>Understand the typical range of your sales metrics</li>
                    <li>Identify common and rare values</li>
                    <li>Spot outliers or unusual patterns</li>
                    <li>Assess the spread and concentration of your data</li>
                  </ul>
                </div>
              </div>
            </Paper>
          </Grid>
        )}

        {data && (
          <Grid item xs={12}>
            <Paper className={classes.paper}>
              <Typography variant="h6" className={classes.title}>
                Summary Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography variant="h6">Total {selectedMetric}</Typography>
                  <Typography variant="h4">
                    {data.reduce((sum, d) => sum + d[selectedMetric], 0).toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    The sum of all {selectedMetric} values in the selected period
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="h6">Average {selectedMetric}</Typography>
                  <Typography variant="h4">
                    {(data.reduce((sum, d) => sum + d[selectedMetric], 0) / data.length).toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    The mean value of {selectedMetric} per time period
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="h6">Max {selectedMetric}</Typography>
                  <Typography variant="h4">
                    {Math.max(...data.map(d => d[selectedMetric])).toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    The highest value of {selectedMetric} achieved
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="h6">Min {selectedMetric}</Typography>
                  <Typography variant="h4">
                    {Math.min(...data.map(d => d[selectedMetric])).toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    The lowest value of {selectedMetric} recorded
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </div>
  );
}

export default SalesAnalysis; 