const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());

app.use(bodyParser.json());

app.post('/predict', (req, res) => {
  const inputData = req.body.inputData;

  // Run a Python script as a subprocess to make predictions
  const pythonProcess = spawn('python', [
    path.join(__dirname, 'predict.py'), // Path to Python script
    JSON.stringify(inputData), // Pass inputData as a JSON string
  ]);

  // Collect predictions from the Python script
  let predictions = '';
  pythonProcess.stdout.on('data', (data) => {
    predictions += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
      try {
        const result = JSON.parse(predictions);
        res.json({ predictions: result });
      } catch (error) {
        res.status(500).json({ error: 'Prediction failed' });
      }
    } else {
      res.status(500).json({ error: 'Prediction process exited with an error' });
    }
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
