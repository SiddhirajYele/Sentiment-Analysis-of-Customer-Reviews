const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process'); // <-- native Node module
const app = express();
const port = 5001;

app.use(cors());
app.use(express.json());

app.post('/analyze', (req, res) => {
  const { review } = req.body;
  console.log('✅ Received review:', review);

  if (!review) {
    return res.status(400).json({ error: 'Review text is required' });
  }

  const pythonProcess = spawn('C:\\Python313\\python.exe', ['predict.py', review]);

  let output = '';
  let errorOutput = '';

  // Collect stdout data
  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  // Collect stderr data (errors)
  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  // Handle process end
  pythonProcess.on('close', (code) => {
    if (code !== 0 || errorOutput) {
      console.error('❌ Python script error:', errorOutput);
      return res.status(500).json({ error: 'Python error: ' + errorOutput });
    }

    console.log('📦 Raw Python output:', output);

    try {
      const [sentiment, score] = output.trim().split(',');
      res.json({ sentiment, score: parseFloat(score) });
    } catch (e) {
      console.error('❌ Failed to parse Python output:', output);
      res.status(500).json({ error: 'Failed to parse Python output' });
    }
  });
});

app.listen(port, () => {
  console.log(`✅ Backend server running on http://localhost:${port}`);
});
