import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:3001/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div>
            <h1>Deep Learning Model</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFileChange} />
                <button type="submit">Submit</button>
            </form>
            {prediction && <div>Prediction: {prediction}</div>}
        </div>
    );
};

export default App;
