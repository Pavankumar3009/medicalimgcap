require("dotenv").config();
const express = require("express");
const app = express();
const cors = require("cors");
const router = require("./Routes/router");
const mongoose = require("mongoose");

const { Readable } = require('stream');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
    res.send("working???");
});

// middleware
app.use(express.json());
app.use(cors());
app.use(router);

const PORT = process.env.PORT || 3020;
const DB = process.env.DATABASE;

mongoose.connect(DB, {
    // useUnifiedTopology:true,
    // useNewUrlParser:true
}).then(() => console.log("Database Connected"))
.catch((error) => {
    console.log("error", error);
});

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.post('/predictdl', upload.single('file'), async (req, res) => {
    try {
        const formData = new FormData();
        console.log("kanisam try block loki ayina vachhindhi");
        
        // Append file buffer to FormData
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });
       
        const response = await axios.post('http://localhost:5000/predictdl', formData, {
            headers: {
                ...formData.getHeaders(),
            },
        });

        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.post('/predictllm', upload.single('file'), async (req, res) => {
    try {
        const formData = new FormData();

        // Append file buffer to FormData
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post('http://localhost:5000/predictllm', formData, {
            headers: {
                ...formData.getHeaders(),
            },
        });

        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
