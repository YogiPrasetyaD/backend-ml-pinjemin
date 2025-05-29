import express from 'express';
import axios from 'axios';
import cors from 'cors'; // <-- Import middleware cors

const app = express();
const PORT = process.env.PORT || 5001; // <-- Pastikan ini 5001 untuk menghindari konflik dengan backend utama
const FASTAPI_URL = 'http://127.0.0.1:8000';

// Gunakan middleware cors SEBELUM rute-rute Anda
// Ini akan mengizinkan permintaan dari semua origin (*)
app.use(cors());

app.use(express.json());


// POST /api/recommend/user
app.post('/api/recommend/user', async (req, res) => {
  try {
    console.log(`Proxying /recommend/user request to FastAPI: ${FASTAPI_URL}/recommend/user with body:`, req.body); // Tambahkan logging
    const response = await axios.post(`${FASTAPI_URL}/recommend/user`, req.body);
    res.json(response.data);
  } catch (err) {
    console.error('Error proxying /api/recommend/user:', err.message); // Logging error yang lebih jelas
    // Periksa apakah ini error axios dengan response dari FastAPI
    if (err.response) {
        console.error('FastAPI response data:', err.response.data);
        console.error('FastAPI response status:', err.response.status);
        // Teruskan status dan body error dari FastAPI jika ada
        res.status(err.response.status).json(err.response.data);
    } else {
        res.status(500).json({ error: 'Failed to fetch user recommendation or internal error' });
    }
  }
});

// POST /api/recommend/item
app.post('/api/recommend/item', async (req, res) => {
  try {
     console.log(`Proxying /recommend/item request to FastAPI: ${FASTAPI_URL}/recommend/item with body:`, req.body); // Tambahkan logging
    const response = await axios.post(`${FASTAPI_URL}/recommend/item`, req.body);
    res.json(response.data);
  } catch (err) {
    console.error('Error proxying /api/recommend/item:', err.message); // Logging error yang lebih jelas
     if (err.response) {
        console.error('FastAPI response data:', err.response.data);
        console.error('FastAPI response status:', err.response.status);
        res.status(err.response.status).json(err.response.data);
    } else {
        res.status(500).json({ error: 'Failed to fetch item recommendation or internal error' });
    }
  }
});

// POST /api/recommend/search
app.post('/api/recommend/search', async (req, res) => {
  try {
     console.log(`Proxying /recommend/search request to FastAPI: ${FASTAPI_URL}/recommend/search with body:`, req.body); // Tambahkan logging
    const response = await axios.post(`${FASTAPI_URL}/recommend/search`, req.body);
    res.json(response.data);
  } catch (err) {
     console.error('Error proxying /api/recommend/search:', err.message); // Logging error yang lebih jelas
     if (err.response) {
        console.error('FastAPI response data:', err.response.data);
        console.error('FastAPI response status:', err.response.status);
        res.status(err.response.status).json(err.response.data);
    } else {
        res.status(500).json({ error: 'Failed to fetch search recommendation or internal error' });
    }
  }
});

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}`);
});