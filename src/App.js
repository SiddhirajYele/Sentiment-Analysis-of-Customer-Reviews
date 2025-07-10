import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    console.log('Sending:', review);
    try {
      const response = await axios.post('http://localhost:5001/analyze', { review }, { timeout: 10000000 }); // 10s timeout
      console.log('Received:', response.data);
      setResult(response.data);
    } catch (err) {
      console.error('Request Error:', err.message, err.response ? err.response.data : 'No response');
      setError(`Error: ${err.message}${err.response ? ' - ' + JSON.stringify(err.response.data) : ''}`);
    }
  };

  return (
    <div className="App">
      <h1>Sentiment Analysis of Customer Reviews</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder="Enter your review here..."
          rows="5"
          cols="50"
        />
        <br />
        <button type="submit">Analyze Sentiment</button>
      </form>

      {result && (
        <div className="result">
          <h3>Result:</h3>
          <p>Sentiment: <strong>{result.sentiment}</strong></p>
          <p>Score: {result.score.toFixed(2)}</p>
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;