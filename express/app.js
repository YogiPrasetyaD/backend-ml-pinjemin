import express from "express"
import axios from "axios"
import cors from "cors"

const app = express()
const PORT = process.env.PORT || 5001
const FASTAPI_URL = "http://127.0.0.1:8000"

app.use(cors())
app.use(express.json())

// POST /api/recommend/user
app.post("/api/recommend/user", async (req, res) => {
  try {
    console.log(`Proxying /recommend/user request to FastAPI: ${FASTAPI_URL}/recommend/user with body:`, req.body)
    const response = await axios.post(`${FASTAPI_URL}/recommend/user`, req.body)
    res.json(response.data)
  } catch (err) {
    console.error("Error proxying /api/recommend/user:", err.message)
    if (err.response) {
      console.error("FastAPI response data:", err.response.data)
      console.error("FastAPI response status:", err.response.status)
      res.status(err.response.status).json(err.response.data)
    } else {
      res.status(500).json({ error: "Failed to fetch user recommendation or internal error" })
    }
  }
})

// POST /api/recommend/item
app.post("/api/recommend/item", async (req, res) => {
  try {
    console.log(`Proxying /recommend/item request to FastAPI: ${FASTAPI_URL}/recommend/item with body:`, req.body)

    // ✅ PASTIKAN user_id ada dalam request body
    if (!req.body.user_id) {
      return res.status(400).json({
        error: "user_id is required for item recommendations",
        detail: "Please provide user_id in request body",
      })
    }

    // ✅ PASTIKAN product_id ada dalam request body
    if (!req.body.product_id) {
      return res.status(400).json({
        error: "product_id is required for item recommendations",
        detail: "Please provide product_id in request body",
      })
    }

    const response = await axios.post(`${FASTAPI_URL}/recommend/item`, req.body)
    res.json(response.data)
  } catch (err) {
    console.error("Error proxying /api/recommend/item:", err.message)
    if (err.response) {
      console.error("FastAPI response data:", err.response.data)
      console.error("FastAPI response status:", err.response.status)
      res.status(err.response.status).json(err.response.data)
    } else {
      res.status(500).json({ error: "Failed to fetch item recommendation or internal error" })
    }
  }
})

// POST /api/recommend/search
app.post("/api/recommend/search", async (req, res) => {
  try {
    console.log(`Proxying /recommend/search request to FastAPI: ${FASTAPI_URL}/recommend/search with body:`, req.body)

    // ✅ PASTIKAN user_id ada dalam request body
    if (!req.body.user_id) {
      return res.status(400).json({
        error: "user_id is required for search recommendations",
        detail: "Please provide user_id in request body",
      })
    }

    // ✅ PASTIKAN keyword ada dalam request body
    if (!req.body.keyword) {
      return res.status(400).json({
        error: "keyword is required for search recommendations",
        detail: "Please provide keyword in request body",
      })
    }

    const response = await axios.post(`${FASTAPI_URL}/recommend/search`, req.body)
    res.json(response.data)
  } catch (err) {
    console.error("Error proxying /api/recommend/search:", err.message)
    if (err.response) {
      console.error("FastAPI response data:", err.response.data)
      console.error("FastAPI response status:", err.response.status)
      res.status(err.response.status).json(err.response.data)
    } else {
      res.status(500).json({ error: "Failed to fetch search recommendation or internal error" })
    }
  }
})

// POST /api/refresh_data
app.post("/api/refresh_data", async (req, res) => {
  try {
    console.log(`Proxying /refresh_data request to FastAPI: ${FASTAPI_URL}/refresh_data`)
    const response = await axios.post(`${FASTAPI_URL}/refresh_data`, req.body)
    res.status(response.status).json(response.data)
  } catch (err) {
    console.error("Error proxying /api/refresh_data:", err.message)
    if (err.response) {
      res.status(err.response.status).json(err.response.data)
    } else {
      res.status(500).json({ error: "Failed to refresh data or internal proxy error" })
    }
  }
})

app.listen(PORT, () => {
  console.log(`Express server running at http://localhost:${PORT}`)
})
