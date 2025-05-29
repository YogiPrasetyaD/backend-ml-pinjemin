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

// --- TAMBAHKAN ROUTE BARU UNTUK REFRESH DATA ---
app.post('/api/refresh_data', async (req, res) => {
  try {
    console.log(`Proxying /refresh_data request to FastAPI: ${FASTAPI_URL}/refresh_data`);
    // Teruskan permintaan POST ke backend FastAPI ML
    // Gunakan axios.post tanpa body jika refresh_data endpoint tidak memerlukan body
    const response = await axios.post(`${FASTAPI_URL}/refresh_data`, req.body); // Teruskan body jika ada, atau gunakan {} jika tidak perlu body

    // Kirimkan respons dari FastAPI kembali ke frontend
    res.status(response.status).json(response.data);
  } catch (err) {
    console.error('Error proxying /api/refresh_data:', err.message);
    // Handle error, kirim status error ke frontend
    if (err.response) {
      // Error dari respons FastAPI
      res.status(err.response.status).json(err.response.data);
    } else {
      // Error koneksi atau error lain
      res.status(500).json({ error: 'Failed to refresh data or internal proxy error' });
    }
  }
});
// --- AKHIR ROUTE BARU ---

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}`);
});