from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import mysql.connector
from decimal import Decimal, InvalidOperation as DecimalException

app = FastAPI()

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost") # Default to localhost if not set
DB_USER = os.getenv("DB_USER", "root")      # Default to root if not set
DB_PASSWORD = os.getenv("DB_PASSWORD", "")      # Default to empty if not set
DB_NAME = os.getenv("DB_NAME", "pinjemin")  # Default to pinjemin if not set
DB_PORT = os.getenv("DB_PORT", "3306")      # Default to 3306 if not set

db_connection = None

def get_db_connection():
    """Establishes and returns a database connection."""
    global db_connection
    if db_connection is None or not (db_connection and db_connection.is_connected()):
        try:
            print(f"Attempting to connect to database: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
            db_connection = mysql.connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=int(DB_PORT)
            )
            print("Database connection established successfully!")
        except mysql.connector.Error as err:
            print(f"Database connection failed: {err}")
            db_connection = None
    return db_connection

def load_product_metadata_from_db():
    """Loads product data from the database and returns a list of text features for TF-IDF."""
    global product_data, product_id_to_index, index_to_product_id
    product_data = {}
    product_id_to_index = {}
    index_to_product_id = {}
    product_texts = [] # List to hold text data for TF-IDF

    db = get_db_connection()
    if db is None:
        print("Cannot load product data: Database connection failed.")
        return product_texts # Return empty list on failure

    cursor = None
    try:
        cursor = db.cursor(dictionary=True)
        # Fetch 'description' column as well if needed for TF-IDF
        query = "SELECT id AS product_id, name AS product_name, user_id AS seller_id, price_sell AS product_price, description FROM items" # Added description
        print(f"Executing query: {query}")
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Loaded {len(results)} products from database.")

        def safe_float(val, default=0.0):
            try:
                if isinstance(val, Decimal): return float(val)
                if val is None: return default
                return float(val)
            except (ValueError, TypeError, DecimalException):
                return default

        for idx, row in enumerate(results):
            product_id = int(row["product_id"])
            product_data[idx] = {
                "product_name": row["product_name"],
                "seller_id": int(row["seller_id"]),
                "product_rating": safe_float(row.get("product_rating")), # Will likely be None unless added to DB
                "product_price": safe_float(row.get("product_price")),
                # Combine name and description for text features, handle None
                "text_features": f"{row.get('product_name', '')} {row.get('description', '') or ''}".strip() # Use name + description
            }
            product_id_to_index[product_id] = idx
            index_to_product_id[idx] = product_id
            # Add the combined text to the list for TF-IDF
            product_texts.append(product_data[idx]["text_features"])

        print("Product metadata loaded successfully from database.")
        return product_texts # Return the list of texts

    except mysql.connector.Error as err:
        print(f"Error loading product data from database: {err}")
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred during data loading: {e}")
    finally:
        if cursor:
            cursor.close()
    return product_texts # Return whatever was loaded, potentially empty list

# Optional: Connect on startup, load data, and regenerate TF-IDF
@app.on_event("startup") # Modified startup_event
async def startup_event():
    get_db_connection() # Connect to DB
    # Load product data and get text features list
    product_texts = load_product_metadata_from_db()

    # Regenerate TF-IDF matrix from the loaded text data using the loaded vectorizer
    global tfidf_matrix # Declare global to modify
    if product_texts and vectorizer:
        try:
            # Transform the text data loaded from the DB into a new TF-IDF matrix
            tfidf_matrix = vectorizer.transform(product_texts)
            print(f"Regenerated TF-IDF matrix with shape: {tfidf_matrix.shape}")
        except Exception as e:
            print(f"Error regenerating TF-IDF matrix: {e}")
            tfidf_matrix = None # Set to None if regeneration fails
            # If tfidf_matrix is None, search and item recommendations will likely fail later.
            # You might want to add checks in endpoints for tfidf_matrix being None.


# Optional: Close connection on shutdown
@app.on_event("shutdown")
def shutdown_event():
    global db_connection
    if db_connection and db_connection.is_connected():
        db_connection.close()
        print("Database connection closed.")

# --- End Database Configuration ---

model_user = tf.keras.models.load_model("models/model.h5", compile=False)
model_search = tf.keras.models.load_model("models/model_final.h5", compile=False)
model_item = tf.keras.models.load_model("models/recommendation_model.h5", compile=False)

vectorizer = joblib.load("models/vectorizer.pkl")
tfidf_matrix = None

NUM_CATEGORIES = 10

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10

class SearchRequest(BaseModel):
    keyword: str
    top_n: int = 10

class ProductRecommendation(BaseModel):
    product_id: int
    product_name: str = None
    seller_id: int = None
    product_rating: float = None
    product_price: float = None
    score: float = None

class RecommendationResponse(BaseModel):
    recommendations: List[ProductRecommendation]

def build_recommendations(scores: np.ndarray, top_n: int) -> List[ProductRecommendation]:
   
    top_indices = scores.argsort()[-top_n:][::-1]
    recs = []
    for idx in top_indices:
        if idx < 0 or idx >= len(product_data):
             print(f"Warning: Invalid product index encountered: {idx}. Skipping recommendation for this index.")
             continue

        info = product_data.get(idx, {})
        product_id_val = index_to_product_id.get(idx, idx)

        recs.append(ProductRecommendation(
            product_id=product_id_val,
            score=float(scores[idx]) if idx < len(scores) else None,
            product_name=info.get("product_name"),
            seller_id=info.get("seller_id"),
            product_rating=info.get("product_rating"),
            product_price=info.get("product_price"),
        ))
    return recs

# Endpoint user-based recommendation
@app.post("/recommend/user", response_model=RecommendationResponse)
def recommend_user(req: RecommendationRequest):
    global product_data
    if not product_data:
        print("Product data not loaded on endpoint call, attempting to load...")
        load_product_metadata_from_db()
        if not product_data:
             raise HTTPException(status_code=500, detail="Product data could not be loaded from database.")

    user_vector = np.array([[req.user_id]])
    scores = model_user.predict(user_vector, verbose=0)[0]

    expected_score_len = len(product_data)
    if model_user.output_shape[-1] != expected_score_len:
         print(f"Warning: User model output shape ({model_user.output_shape[-1]}) does not match product data size ({expected_score_len}). Recommendations may be incorrect.")
         if scores.shape[0] > expected_score_len:
             scores = scores[:expected_score_len]
             print(f"Warning: Truncated scores to match product data size: {scores.shape[0]}")
         elif scores.shape[0] < expected_score_len:
              raise HTTPException(status_code=500, detail=f"User model output shape ({model_user.output_shape[-1]}) is smaller than product data size ({expected_score_len}). Cannot generate recommendations.")

    recommendations = build_recommendations(scores, req.top_n)
    return RecommendationResponse(recommendations=recommendations)

# Endpoint search-based recommendation
@app.post("/recommend/search", response_model=RecommendationResponse)
def recommend_search(req: SearchRequest):
    global product_data, tfidf_matrix # Declare globals
    if not product_data or tfidf_matrix is None: # Check both
        print("Product data or TF-IDF matrix not loaded/regenerated on endpoint call, attempting to load...")
        # Re-attempt loading might not regenerate tfidf_matrix unless startup_event is re-run
        # A better approach might be to raise HTTPException directly if product_data or tfidf_matrix is None
        load_product_metadata_from_db() # This will load product_data but not regenerate tfidf_matrix here
        # If product_data is loaded, try regenerating tfidf_matrix if it's None
        if not product_data or tfidf_matrix is None:
             if product_data and vectorizer and tfidf_matrix is None:
                 try:
                     print("Attempting to regenerate TF-IDF matrix during endpoint call.")
                     # Need product_texts again, maybe store them globally or refetch?
                     # Refetching here is inefficient. Relying on startup load is better.
                     # Let's simplify: if tfidf_matrix is None, it's likely a startup failure.
                     raise HTTPException(status_code=500, detail="Product data or TF-IDF matrix could not be loaded/regenerated.")
                 except Exception as e:
                      raise HTTPException(status_code=500, detail=f"Product data or TF-IDF matrix could not be loaded/regenerated: {e}")
             else:
                 raise HTTPException(status_code=500, detail="Product data or TF-IDF matrix could not be loaded/regenerated.")

    expected_tfidf_rows = len(product_data)
    if tfidf_matrix.shape[0] != expected_tfidf_rows:
         print(f"Warning: TF-IDF matrix size ({tfidf_matrix.shape[0]} rows) does not match product data size ({expected_tfidf_rows}). Search may be incorrect.")
         if tfidf_matrix.shape[0] > len(product_data):
             print(f"Warning: Truncating TF-IDF matrix to {expected_tfidf_rows} rows.")

    query_vec = vectorizer.transform([req.keyword.lower()])

    if tfidf_matrix.shape[0] > len(product_data):
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix[:len(product_data)]).flatten()
    else:
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    candidate_indices = cosine_sim.argsort()[-50:][::-1]

    hasil = []

    for idx in candidate_indices:
        if idx < 0 or idx >= len(product_data):
            print(f"Warning: Skipping invalid candidate index: {idx} during search processing.")
            continue

        info = product_data.get(idx, {})
        category_one_hot = np.zeros(NUM_CATEGORIES)
        category_label_val = info.get("category_label", None)
        if category_label_val is not None:
             try:
                 category_label = int(category_label_val)
                 if 0 <= category_label < NUM_CATEGORIES:
                    category_one_hot[category_label] = 1
             except (ValueError, TypeError):
                 print(f"Warning: Could not process category_label '{category_label_val}' for index {idx}.")
                 pass

        expected_input_dim = model_search.input_shape[-1]
        tfidf_len = expected_input_dim - NUM_CATEGORIES

        if idx < tfidf_matrix.shape[0]:
            produk_tfidf_vec = tfidf_matrix[idx].toarray().flatten()
            if produk_tfidf_vec.shape[0] > tfidf_len:
                produk_tfidf_vec = produk_tfidf_vec[:tfidf_len]
            elif produk_tfidf_vec.shape[0] < tfidf_len:
                 pad_width = tfidf_len - produk_tfidf_vec.shape[0]
                 produk_tfidf_vec = np.pad(produk_tfidf_vec, (0, pad_width), mode='constant')
        else:
            print(f"Warning: Index {idx} out of bounds for tfidf_matrix. Cannot process search for this candidate.")
            continue

        x_input = np.concatenate([category_one_hot, produk_tfidf_vec]).reshape(1, -1)

        if x_input.shape[-1] != expected_input_dim:
            print(f"Error: Input shape for model_search ({x_input.shape[-1]}) does not match expected shape ({expected_input_dim}). Skipping product {idx}.")
            continue

        prob = model_search.predict(x_input, verbose=0)[0][0]
        hasil.append((prob, idx))

    hasil_sorted = sorted(hasil, key=lambda x: x[0], reverse=True)[:req.top_n]
    recommendations = []
    for score, idx in hasil_sorted:
        if idx < 0 or idx >= len(product_data):
             print(f"Warning: Skipping invalid recommended index after search sorting: {idx}.")
             continue

        info = product_data.get(idx, {})
        product_id_val = index_to_product_id.get(idx, idx)

        recommendations.append(ProductRecommendation(
            product_id=product_id_val,
            score=float(score),
            product_name=info.get("product_name"),
            seller_id=info.get("seller_id"),
            product_rating=info.get("product_rating"),
            product_price=info.get("product_price"),
        ))
    return RecommendationResponse(recommendations=recommendations)


class ItemRequest(BaseModel):
    product_id: int
    top_n: int = 10


@app.post("/recommend/item", response_model=RecommendationResponse)
def recommend_item(req: ItemRequest):
    global product_data, tfidf_matrix # Declare globals
    if not product_data or tfidf_matrix is None: # Check both
        print("Product data or TF-IDF matrix not loaded/regenerated on endpoint call, attempting to load...")
        load_product_metadata_from_db() # Loads product_data, doesn't regenerate tfidf_matrix
        if not product_data or tfidf_matrix is None:
             if product_data and vectorizer and tfidf_matrix is None:
                 try:
                     print("Attempting to regenerate TF-IDF matrix during endpoint call.")
                     # Same inefficiency issue as search endpoint. Rely on startup.
                     raise HTTPException(status_code=500, detail="Product data or TF-IDF matrix could not be loaded/regenerated.")
                 except Exception as e:
                      raise HTTPException(status_code=500, detail=f"Product data or TF-IDF matrix could not be loaded/regenerated: {e}")
             else:
                 raise HTTPException(status_code=500, detail="Product data or TF-IDF matrix could not be loaded/regenerated.")

    if req.product_id not in product_id_to_index:
        print(f"Warning: Product ID {req.product_id} not found in product_id_to_index mapping. Cannot generate item recommendations.")
        return RecommendationResponse(recommendations=[])

    query_index = product_id_to_index[req.product_id]

    X_all = tfidf_matrix.toarray()

    expected_data_rows = len(product_data)
    if X_all.shape[0] != expected_data_rows:
         print(f"Warning: X_all (derived from tfidf_matrix) size ({X_all.shape[0]} rows) does not match product data size ({expected_data_rows}). Item recommendations may be incorrect.")
         if X_all.shape[0] > expected_data_rows:
             print(f"Warning: Truncating X_all to {expected_data_rows} rows.")
             X_all = X_all[:expected_data_rows, :]

    expected_item_input_dim = model_item.input_shape[-1]
    if X_all.shape[1] > expected_item_input_dim:
        print(f"Warning: Truncating X_all columns ({X_all.shape[1]} -> {expected_item_input_dim}) to match model_item input shape.")
        X_all = X_all[:, :expected_item_input_dim]
    elif X_all.shape[1] < expected_item_input_dim:
        print(f"Warning: Padding X_all columns ({X_all.shape[1]} -> {expected_item_input_dim}) to match model_item input shape.")
        pad_width = expected_item_input_dim - X_all.shape[1]
        X_all = np.pad(X_all, ((0,0),(0,pad_width)), mode='constant')

    if query_index < 0 or query_index >= X_all.shape[0]:
         print(f"Error: Query index {query_index} out of bounds for X_all (shape: {X_all.shape}). Cannot predict item embedding.")
         raise HTTPException(status_code=500, detail="Internal error: Invalid query index for item processing.")

    item_embeddings = model_item.predict(X_all, verbose=0)

    if query_index < 0 or query_index >= item_embeddings.shape[0]:
         print(f"Error: Query index {query_index} out of bounds for item_embeddings (shape: {item_embeddings.shape}). Cannot calculate similarities.")
         raise HTTPException(status_code=500, detail="Internal error: Invalid query index for item embeddings.")

    if len(item_embeddings.shape) == 2:
        query_embedding = item_embeddings[query_index].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, item_embeddings)[0]

    else:
        print(f"Warning: Unexpected item_embeddings shape: {item_embeddings.shape}. Cannot calculate cosine similarity.")
        raise HTTPException(status_code=500, detail="Internal error: Unexpected model output shape for item embeddings.")

    similarities[query_index] = -np.inf

    similar_indices = similarities.argsort()[::-1][:req.top_n]

    recs = []
    for idx in similar_indices:
        if idx < 0 or idx >= len(product_data):
             print(f"Warning: Skipping invalid recommended index after item similarity sorting: {idx}.")
             continue

        pid = index_to_product_id.get(idx, None)
        if pid is None:
             print(f"Warning: No product ID found for index {idx}. Skipping recommendation.")
             continue

        info = product_data.get(idx, {})

        recs.append(ProductRecommendation(
            product_id=pid,
            score=float(similarities[idx]),
            product_name=info.get("product_name"),
            seller_id=info.get("seller_id"),
            product_rating=info.get("product_rating"),
            product_price=info.get("product_price"),
        ))
    return RecommendationResponse(recommendations=recs)
