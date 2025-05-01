import React, { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { AppBar, Tabs, Tab, Box, Container, Typography, Paper, Grid } from '@material-ui/core';
import SalesAnalysis from './components/SalesAnalysis';
import MLPredictions from './components/MLPredictions';
import DeepLearning from './components/DeepLearning';
import QuantumComputing from './components/QuantumComputing';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
    backgroundColor: '#f5f7fa',
    minHeight: '100vh',
  },
  appBar: {
    backgroundColor: '#1a237e',
    boxShadow: 'none',
  },
  tab: {
    textTransform: 'none',
    fontWeight: 500,
    fontSize: '1rem',
  },
  container: {
    padding: theme.spacing(4),
  },
  paper: {
    padding: theme.spacing(3),
    marginBottom: theme.spacing(3),
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  },
  title: {
    marginBottom: theme.spacing(4),
    color: '#1a237e',
    fontWeight: 600,
  },
  subtitle: {
    color: '#666',
    marginBottom: theme.spacing(2),
  },
}));

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box p={3}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const classes = useStyles();
  const [value, setValue] = useState(0);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  return (
    <div className={classes.root}>
      <AppBar position="static" className={classes.appBar}>
        <Tabs
          value={value}
          onChange={handleChange}
          indicatorColor="primary"
          textColor="inherit"
          variant="fullWidth"
        >
          <Tab label="Sales Analysis" className={classes.tab} />
          <Tab label="ML Predictions" className={classes.tab} />
          <Tab label="Deep Learning" className={classes.tab} />
          <Tab label="Quantum Computing" className={classes.tab} />
        </Tabs>
      </AppBar>

      <Container maxWidth="lg" className={classes.container}>
        <Paper className={classes.paper}>
          <Typography variant="h4" className={classes.title}>
            Advanced Analytics Dashboard
          </Typography>
          <Typography variant="subtitle1" className={classes.subtitle}>
            Explore comprehensive data analysis, machine learning predictions, deep learning models, and quantum computing simulations
          </Typography>
        </Paper>

        <TabPanel value={value} index={0}>
          <SalesAnalysis />
        </TabPanel>
        <TabPanel value={value} index={1}>
          <MLPredictions />
        </TabPanel>
        <TabPanel value={value} index={2}>
          <DeepLearning />
        </TabPanel>
        <TabPanel value={value} index={3}>
          <QuantumComputing />
        </TabPanel>
      </Container>
    </div>
  );
}

export default App;
