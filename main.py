from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from datetime import datetime, timedelta
import logging
import os
import json
from difflib import SequenceMatcher
import random
import hashlib
# Add SQLAlchemy imports
from sqlalchemy import create_engine, text
import joblib
from scipy.sparse import load_npz

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Recommendation System", version="6.0.0")

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "pinjemin"),
    "port": int(os.getenv("DB_PORT", "3306"))
}

# Category Mapping
CATEGORY_MAP = {
    1: {
        "name": "Masak",
        "keywords": ["masak", "dapur", "telenan", "talenan", "panci", "wajan", "spatula", 
                     "blender", "kompor", "kulkas", "pisau", "sendok", "resep", "makanan", "cooking"]
    },
    2: {
        "name": "Fotografi", 
        "keywords": ["fotografi", "kamera", "camera", "foto", "lensa", "tripod", "flash",
                     "canon", "nikon", "sony", "portrait", "landscape", "editing", "photography"]
    },
    3: {
        "name": "Membaca",
        "keywords": ["membaca", "buku", "novel", "komik", "majalah", "ebook", "reading",
                     "literature", "fiction", "textbook", "author", "writer", "book"]
    }
}

# Load models and data
model_user = tf.keras.models.load_model("models/model.h5", compile=False)
model_search = tf.keras.models.load_model("models/model_final.h5", compile=False)
model_item = tf.keras.models.load_model("models/recommendation_model.h5", compile=False)
vectorizer = joblib.load("models/vectorizer.pkl")
tfidf_matrix = load_npz("models/tfidf_matrix.npz")
df = pd.read_csv("models/all_dataset.csv").reset_index(drop=True)

product_data = {}
product_id_to_index = {}
index_to_product_id = {}

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

for idx, row in df.iterrows():
    product_id = int(row["product_id"])
    product_data[idx] = {
        "product_name": row["product_name"],
        "seller_id": int(row["seller_id"]),
        "product_rating": safe_float(row.get("product_rating")),
        "product_price": safe_float(row.get("product_price")),
    }
    product_id_to_index[product_id] = idx
    index_to_product_id[idx] = product_id

NUM_CATEGORIES = 10

class ModelManager:
    """Manages pre-trained models"""
    
    def __init__(self):
        self.model_user = None
        self.model_search = None
        self.model_item = None
        self.models_loaded = False
        self.user_similarity_matrix = None
        self.last_refresh = None
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists("models/model.h5"):
                self.model_user = tf.keras.models.load_model("models/model.h5", compile=False)
                logger.info("✅ User model loaded")
            
            if os.path.exists("models/model_final.h5"):
                self.model_search = tf.keras.models.load_model("models/model_final.h5", compile=False)
                logger.info("✅ Search model loaded")
            
            if os.path.exists("models/recommendation_model.h5"):
                self.model_item = tf.keras.models.load_model("models/recommendation_model.h5", compile=False)
                logger.info("✅ Item model loaded")
            
            self._generate_user_similarity_matrix()
            
            self.models_loaded = True
            self.last_refresh = datetime.now()
            logger.info("✅ All available models loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not load some models: {e}")
            self.models_loaded = False
            return False
    
    def _generate_user_similarity_matrix(self):
        """Generate user similarity matrix based on purchase history"""
        try:
            engine = DatabaseManager.get_engine()
            if not engine:
                logger.warning("Cannot generate user similarity matrix: No database connection")
                return
            
            query = """
            SELECT 
                t.buyer_id as user_id,
                t.item_id,
                i.category_id
            FROM transactions t
            JOIN items i ON t.item_id = i.id
            ORDER BY t.buyer_id, t.created_at DESC
            """
            
            df = pd.read_sql_query(text(query), engine)
            if df.empty:
                logger.warning("No transaction data found for similarity matrix")
                return
            
            user_item_matrix = pd.crosstab(df['user_id'], df['item_id'])
            user_item_matrix = user_item_matrix.fillna(0)
            
            if len(user_item_matrix) > 1:
                self.user_similarity_matrix = pd.DataFrame(
                    cosine_similarity(user_item_matrix),
                    index=user_item_matrix.index,
                    columns=user_item_matrix.index
                )
                logger.info(f"✅ Generated user similarity matrix for {len(self.user_similarity_matrix)} users")
            else:
                logger.warning("Not enough users for similarity matrix")
                self.user_similarity_matrix = None
                
        except Exception as e:
            logger.error(f"Error generating user similarity matrix: {e}")
            self.user_similarity_matrix = None
    
    def get_similar_users(self, user_id, top_n=5):
        """Get similar users based on similarity matrix"""
        if self.user_similarity_matrix is None or user_id not in self.user_similarity_matrix.index:
            return DatabaseManager.get_similar_users(user_id, top_n)
        
        user_similarities = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False)
        user_similarities = user_similarities[user_similarities.index != user_id]
        top_similar = user_similarities.head(top_n)
        
        result = pd.DataFrame({
            'similar_user_id': top_similar.index,
            'common_items': top_similar.values * 10
        })
        
        logger.info(f"Found {len(result)} similar users for user {user_id} using similarity matrix")
        return result
    
    def predict_user_preferences(self, user_features):
        """Predict user preferences using trained model"""
        if self.model_user and self.models_loaded:
            try:
                return self.model_user.predict(user_features, verbose=0)
            except Exception as e:
                logger.error(f"User model prediction failed: {e}")
        return None
    
    def get_item_embeddings(self, item_indices: np.ndarray) -> Optional[np.ndarray]:
        """Generate embeddings for given item indices using model_item.
        FIX: Use TF-IDF vector as input if model_item expects vector, not just index.
        """
        if self.model_item and self.models_loaded:
            try:
                # --- FIX: Use TF-IDF vector as input, not just index ---
                # item_indices: list of row indices in self.products_df
                # Ambil TF-IDF vector untuk semua produk
                tfidf_all = tfidf_matrix.toarray()
                # Pastikan shape sesuai input model_item
                input_dim = self.model_item.input_shape[-1]
                if tfidf_all.shape[1] > input_dim:
                    tfidf_all = tfidf_all[:, :input_dim]
                elif tfidf_all.shape[1] < input_dim:
                    tfidf_all = np.pad(tfidf_all, ((0,0),(0,input_dim-tfidf_all.shape[1])), mode='constant')
                # Ambil hanya baris yang diminta
                input_data = tfidf_all[item_indices]
                embeddings = self.model_item.predict(input_data, verbose=0)
                logger.info(f"✅ Generated embeddings for {len(item_indices)} items (TF-IDF input).")
                return embeddings
            except Exception as e:
                logger.error(f"Item model prediction (get_item_embeddings) failed: {e}")
        return None

    def refresh_models(self):
        """Refresh all models and similarity matrix"""
        try:
            success = self.load_models()
            self._generate_user_similarity_matrix()
            self.last_refresh = datetime.now()
            return success
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            return False

class DatabaseManager:
    """Database connection manager with SQLAlchemy support"""
    
    @staticmethod
    def get_connection_string():
        """Get SQLAlchemy connection string"""
        db_config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'pinjemin')
        }
        return f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
    
    @staticmethod
    def get_engine():
        """Get SQLAlchemy engine"""
        try:
            connection_string = DatabaseManager.get_connection_string()
            return create_engine(connection_string)
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {e}")
            return None
    
    @staticmethod
    def get_connection():
        """Legacy method for direct MySQL connection (fallback)"""
        try:
            db_config = {
                'host': os.environ.get('DB_HOST', 'localhost'),
                'user': os.environ.get('DB_USER', 'root'),
                'password': os.environ.get('DB_PASSWORD', ''),
                'database': os.environ.get('DB_NAME', 'pinjemin')
            }
            return mysql.connector.connect(**db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    @staticmethod
    def load_available_items():
        """Load available items from database"""
        try:
            engine = DatabaseManager.get_engine()
            if not engine:
                return pd.DataFrame()
                
            query = """
            SELECT 
                id as product_id,
                COALESCE(name, '') as product_name,
                COALESCE(description, '') as description,
                COALESCE(price_sell, 0) as price,
                COALESCE(category_id, 1) as category_id,
                COALESCE(user_id, 1) as seller_id,
                created_at
            FROM items 
            WHERE status = 'available'
            ORDER BY id
            """
            
            df = pd.read_sql_query(text(query), engine)
            logger.info(f"Loaded {len(df)} available items from database")
            return df
        except Exception as e:
            logger.warning(f"Failed to load available items: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_user_owned_items(user_id):
        """Get items owned by user (where user is the seller)"""
        try:
            engine = DatabaseManager.get_engine()
            if not engine:
                return set()
                
            query = """
            SELECT id as product_id
            FROM items 
            WHERE user_id = :user_id
            """
            
            df = pd.read_sql_query(text(query), engine, params={"user_id": user_id})
            owned_items = set(df['product_id'].tolist())
            
            logger.info(f"User {user_id} owns {len(owned_items)} items")
            return owned_items
        except Exception as e:
            logger.warning(f"Failed to get user owned items: {e}")
            return set()
    
    @staticmethod
    def get_user_posted_items(user_id):
        """Get items that user has posted"""
        try:
            engine = DatabaseManager.get_engine()
            if not engine:
                return pd.DataFrame()
                
            query = """
            SELECT 
                id as product_id,
                name as product_name,
                category_id,
                description
            FROM items 
            WHERE status = 'available' 
            AND user_id = :user_id
            ORDER BY created_at DESC
            LIMIT 10
            """
            
            df = pd.read_sql_query(text(query), engine, params={"user_id": user_id})
            
            logger.info(f"Found {len(df)} items posted by user {user_id}")
            return df
        except Exception as e:
            logger.warning(f"No user posted items found: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_user_purchase_history(user_id):
        """Get user's purchase history for collaborative filtering"""
        try:
            engine = DatabaseManager.get_engine()
            if not engine:
                return pd.DataFrame()
                
            query = """
            SELECT 
                i.id as product_id,
                i.name as product_name,
                i.category_id,
                i.price_sell as price,
                i.user_id as seller_id,
                t.created_at as purchase_date
            FROM transactions t
            JOIN items i ON t.item_id = i.id
            WHERE t.buyer_id = :user_id
            ORDER BY t.created_at DESC
            """
            
            df = pd.read_sql_query(text(query), engine, params={"user_id": user_id})
            logger.info(f"Found {len(df)} purchase history items for user {user_id}")
            return df
        except Exception as e:
            logger.warning(f"Failed to get user purchase history: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_similar_users(user_id, limit=5):
        """Find similar users based on purchase patterns"""
        try:
            engine = DatabaseManager.get_engine()
            if not engine:
                return pd.DataFrame()
                
            query = """
            SELECT 
                t2.buyer_id as similar_user_id,
                COUNT(*) as common_items
            FROM transactions t1
            JOIN transactions t2 ON t1.item_id = t2.item_id AND t1.buyer_id != t2.buyer_id
            WHERE t1.buyer_id = :user_id
            GROUP BY t2.buyer_id
            ORDER BY common_items DESC
            LIMIT :limit
            """
            
            df = pd.read_sql_query(text(query), engine, params={"user_id": user_id, "limit": limit})
            logger.info(f"Found {len(df)} similar users for user {user_id}")
            return df
        except Exception as e:
            logger.warning(f"Failed to get similar users: {e}")
            return pd.DataFrame()

class FastSemanticEngine:
    """Optimized semantic search engine"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.products_df = None
        self.category_cache = {}
    
    def fit(self, products_df):
        """Fit semantic engine with optimizations"""
        self.products_df = products_df.copy()
        
        self.products_df['search_text'] = (
            self.products_df['product_name'].fillna('').astype(str) + ' ' +
            self.products_df['description'].fillna('').astype(str) + ' ' +
            self.products_df['category_id'].apply(self._get_category_keywords)
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['search_text'])
        logger.info(f"Semantic engine fitted: {self.tfidf_matrix.shape}")
    
    def _get_category_keywords(self, category_id):
        """Cached category keywords"""
        if category_id not in self.category_cache:
            if category_id in CATEGORY_MAP:
                self.category_cache[category_id] = ' '.join(CATEGORY_MAP[category_id]['keywords'])
            else:
                self.category_cache[category_id] = ''
        return self.category_cache[category_id]
    
    def search(self, query, top_n=10, exclude_seller_id=None):
        """Fast semantic search with guaranteed 10 results, only available products."""
        if not self.tfidf_vectorizer or self.tfidf_matrix.shape[0] == 0:
            return self._fallback_search(top_n, exclude_seller_id)
        
        processed_query = self._preprocess_query(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        all_indices = similarities.argsort()[::-1]

        # --- FILTER ONLY AVAILABLE PRODUCTS ---
        available_ids = set(self.products_df[self.products_df.get('status', 'available').str.lower() == 'available']['product_id'].tolist()) \
            if 'status' in self.products_df.columns else set(self.products_df['product_id'].tolist())

        results = []
        seen_products = set()
        for idx in all_indices:
            if len(results) >= top_n:
                break
            product = self.products_df.iloc[idx]
            product_id = product['product_id']
            # Only available products
            if product_id not in available_ids:
                continue
            if exclude_seller_id and product['seller_id'] == exclude_seller_id:
                continue
            if product_id not in seen_products:
                seen_products.add(product_id)
                results.append(self._create_result(product, similarities[idx]))
        if len(results) < top_n:
            category_results = self._category_search(query, top_n - len(results), seen_products, exclude_seller_id)
            # Only available products in fallback
            category_results = [r for r in category_results if r['product_id'] in available_ids]
            results.extend(category_results)
        if len(results) < top_n:
            popular_results = self._popular_search(top_n - len(results), seen_products, exclude_seller_id)
            # Only available products in fallback
            popular_results = [r for r in popular_results if r['product_id'] in available_ids]
            results.extend(popular_results)
        return results[:top_n]
    
    def find_similar_by_name(self, product_name, exclude_id=None, top_n=10, exclude_seller_id=None):
        """Find similar products by name similarity"""
        if not self.tfidf_vectorizer or self.tfidf_matrix.shape[0] == 0:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([product_name])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_n:
                break
                
            product = self.products_df.iloc[idx]
            if exclude_id and product['product_id'] == exclude_id:
                continue
                
            if exclude_seller_id and product['seller_id'] == exclude_seller_id:
                continue
                
            if similarities[idx] > 0.1:
                results.append(self._create_result(product, similarities[idx]))
        
        return results
    
    def _preprocess_query(self, query):
        """Quick query preprocessing"""
        query = query.lower().strip()
        
        corrections = {
            'telenan': 'talenan cutting board masak dapur',
            'talenan': 'telenan cutting board masak dapur',
            'kamera': 'camera fotografi photography',
            'foto': 'fotografi camera photography',
            'buku': 'book membaca reading novel'
        }
        
        for typo, correction in corrections.items():
            if typo in query:
                query += ' ' + correction
        
        return query
    
    def _create_result(self, product, similarity_score):
        """Create standardized result"""
        return {
            'product_id': product['product_id'],
            'similarity_score': float(max(0.1, similarity_score)),
            'product_name': product['product_name'],
            'category_id': product['category_id'],
            'price': product['price'],
            'seller_id': product['seller_id'],
            'category_name': CATEGORY_MAP.get(product['category_id'], {}).get('name', 'Unknown')
        }
    
    def _category_search(self, query, needed, seen_products, exclude_seller_id=None):
        """Category-based fallback search"""
        relevant_categories = self._find_categories(query)
        results = []
        
        for category_id in relevant_categories:
            if len(results) >= needed:
                break
                
            category_filter = (self.products_df['category_id'] == category_id) & \
                              (~self.products_df['product_id'].isin(seen_products))
            
            if exclude_seller_id:
                category_filter = category_filter & (self.products_df['seller_id'] != exclude_seller_id)
                
            category_products = self.products_df[category_filter].head(needed - len(results))
            
            for _, product in category_products.iterrows():
                results.append(self._create_result(product, 0.6))
                seen_products.add(product['product_id'])
        
        return results
    
    def _popular_search(self, needed, seen_products, exclude_seller_id=None):
        """Popular items fallback"""
        results = []
        
        available_filter = ~self.products_df['product_id'].isin(seen_products)
        
        if exclude_seller_id:
            available_filter = available_filter & (self.products_df['seller_id'] != exclude_seller_id)
            
        available_products = self.products_df[available_filter].head(needed)
        
        for _, product in available_products.iterrows():
            results.append(self._create_result(product, 0.4))
        
        return results
    
    def _find_categories(self, query):
        """Find relevant categories quickly"""
        query_words = set(query.lower().split())
        category_scores = []
        
        for cat_id, cat_info in CATEGORY_MAP.items():
            score = len(query_words.intersection(set(cat_info['keywords'])))
            if score > 0:
                category_scores.append((cat_id, score))
        
        category_scores.sort(key=lambda x: x[1], reverse=True)
        return [cat_id for cat_id, _ in category_scores]
    
    def _fallback_search(self, top_n, exclude_seller_id=None):
        """Emergency fallback"""
        if self.products_df is None or len(self.products_df) == 0:
            return []
        
        results = []
        
        available_filter = pd.Series([True] * len(self.products_df))
        
        if exclude_seller_id:
            available_filter = available_filter & (self.products_df['seller_id'] != exclude_seller_id)
            
        available_products = self.products_df[available_filter].head(top_n)
        
        for _, product in available_products.iterrows():
            results.append(self._create_result(product, 0.3))
        
        return results

class SmartRecommendationEngine:
    """Optimized main recommendation engine"""
    
    def __init__(self):
        self.products_df = None
        self.semantic_engine = FastSemanticEngine()
        self.model_manager = ModelManager()
        self.item_similarity_cache = {}
        self.last_update = None
        self.is_initialized = False
    
    def initialize(self):
        """Fast initialization"""
        try:
            logger.info("Initializing recommendation engine...")
            
            self.model_manager.load_models()
            
            self.products_df = DatabaseManager.load_available_items()
            if self.products_df.empty:
                logger.error("No products found in database!")
                return False
            
            self.semantic_engine.fit(self.products_df)
            
            self._precalculate_similarities()
            
            self.last_update = datetime.now()
            self.is_initialized = True
            
            logger.info(f"✅ Engine initialized with {len(self.products_df)} products")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def refresh_data(self):
        """Refresh all data and models"""
        try:
            logger.info("Refreshing recommendation engine data...")
            
            self.model_manager.refresh_models()
            
            self.products_df = DatabaseManager.load_available_items()
            if self.products_df.empty:
                logger.error("No products found in database after refresh!")
                return False
            
            self.semantic_engine.fit(self.products_df)
            
            self._precalculate_similarities()
            
            self.last_update = datetime.now()
            
            logger.info(f"✅ Engine refreshed with {len(self.products_df)} products")
            return True
            
        except Exception as e:
            logger.error(f"❌ Refresh failed: {e}")
            return False
    
    def _precalculate_similarities(self):
        """Pre-calculate item similarities using model_item embeddings."""
        try:
            if self.model_manager.model_item is None:
                logger.warning("⚠️ model_item not loaded. Falling back to TF-IDF similarity.")
                self._precalculate_similarities_tfidf_fallback()
                return

            logger.info("Pre-calculating item similarities using model_item embeddings...")
            
            # --- FIX: Use index from 0 to len(products_df)-1 for embedding ---
            all_product_indices = list(range(len(self.products_df)))
            if not all_product_indices:
                logger.error("No valid product indices to process for similarity calculation.")
                return

            item_embeddings = self.model_manager.get_item_embeddings(all_product_indices)

            if item_embeddings is None or len(item_embeddings) == 0:
                logger.error("❌ Failed to generate item embeddings. Cannot calculate model-based similarity.")
                return

            similarity_matrix = cosine_similarity(item_embeddings)
            
            matrix_index_to_product_id = {i: self.products_df.iloc[i]['product_id'] for i in all_product_indices}

            for i in range(similarity_matrix.shape[0]):
                current_product_id = matrix_index_to_product_id[i]
                
                similarities = similarity_matrix[i]
                top_indices = similarities.argsort()[-21:-1][::-1]
                
                similar_items = []
                for sim_idx in top_indices:
                    similar_product_id = matrix_index_to_product_id[sim_idx]
                    similar_items.append({
                        'product_id': similar_product_id,
                        'similarity_score': float(similarities[sim_idx])
                    })
                
                self.item_similarity_cache[current_product_id] = similar_items

            logger.info(f"✅ Item similarities pre-calculated for {len(self.item_similarity_cache)} items using model_item.")

        except Exception as e:
            logger.error(f"Error pre-calculating model-based similarities: {e}")

    def _precalculate_similarities_tfidf_fallback(self):
        """Fallback method to calculate similarity using TF-IDF if model fails."""
        try:
            if self.semantic_engine.tfidf_matrix is not None:
                similarity_matrix = cosine_similarity(self.semantic_engine.tfidf_matrix)
                for idx, product_id in enumerate(self.products_df['product_id']):
                    similarities = similarity_matrix[idx]
                    top_indices = similarities.argsort()[-21:-1][::-1]
                    similar_items = []
                    for sim_idx in top_indices:
                        if sim_idx != idx:
                            similar_product_id = self.products_df.iloc[sim_idx]['product_id']
                            similar_items.append({
                                'product_id': similar_product_id,
                                'similarity_score': float(similarities[sim_idx])
                            })
                    self.item_similarity_cache[product_id] = similar_items
                logger.info("✅ Item similarities pre-calculated using TF-IDF fallback.")
        except Exception as e:
            logger.error(f"Error in TF-IDF fallback similarity calculation: {e}")
    
    def get_user_recommendations(self, user_id, top_n=10):
        """Get personalized recommendations for each user using collaborative filtering"""
        try:
            user_owned_items = DatabaseManager.get_user_owned_items(user_id)
            
            similar_users = self.model_manager.get_similar_users(user_id)
            
            if not similar_users.empty:
                logger.info(f"Found {len(similar_users)} similar users for user {user_id}")
                return self._get_collaborative_filtering_recommendations(user_id, similar_users, top_n, user_owned_items)
            
            user_posted_items = DatabaseManager.get_user_posted_items(user_id)
            
            if not user_posted_items.empty:
                logger.info(f"User {user_id} has posted {len(user_posted_items)} items, finding similar items")
                return self._get_recommendations_from_user_items(user_posted_items, user_id, top_n, user_owned_items)
            else:
                logger.info(f"User {user_id} is new, generating personalized recommendations")
                return self._get_personalized_new_user_recommendations(user_id, top_n, user_owned_items)
                
        except Exception as e:
            logger.error(f"Error in user recommendations: {e}")
            return self._get_personalized_new_user_recommendations(user_id, top_n, set())
    
    def _get_collaborative_filtering_recommendations(self, user_id, similar_users, top_n, user_owned_items):
        """Get recommendations based on similar users' purchases"""
        try:
            user_history = DatabaseManager.get_user_purchase_history(user_id)
            purchased_ids = set(user_history['product_id'].tolist() if not user_history.empty else [])
            
            exclude_ids = purchased_ids.union(user_owned_items)
            
            user_category_prefs = {}
            if not user_history.empty:
                category_counts = user_history['category_id'].value_counts()
                total_purchases = category_counts.sum()
                for category_id, count in category_counts.items():
                    user_category_prefs[category_id] = count / total_purchases
            
            recommendations = {}
            
            for _, similar_user in similar_users.iterrows():
                similar_user_id = similar_user['similar_user_id']
                similarity_weight = similar_user['common_items'] / 10
                
                engine = DatabaseManager.get_engine()
                if not engine:
                    continue
                    
                query = """
                SELECT 
                    i.id as product_id,
                    i.name as product_name,
                    i.category_id,
                    i.price_sell as price,
                    i.user_id as seller_id,
                    i.description
                FROM transactions t
                JOIN items i ON t.item_id = i.id
                WHERE t.buyer_id = :user_id
                AND i.status = 'available'
                AND i.user_id != :exclude_user_id
                """
                
                similar_user_items = pd.read_sql_query(text(query), engine, params={"user_id": similar_user_id, "exclude_user_id": user_id})
                
                for _, item in similar_user_items.iterrows():
                    product_id = item['product_id']
                    
                    if product_id in exclude_ids:
                        continue
                        
                    category_boost = 1.0
                    if user_category_prefs and item['category_id'] in user_category_prefs:
                        category_boost = 1.0 + user_category_prefs[item['category_id']]
                    
                    if product_id not in recommendations:
                        recommendations[product_id] = {
                            'score': 0,
                            'product_name': item['product_name'],
                            'category_id': item['category_id'],
                            'price': item['price'],
                            'seller_id': item['seller_id'],
                            'description': item['description']
                        }
                    
                    recommendations[product_id]['score'] += similarity_weight * category_boost
            
            if self.semantic_engine.tfidf_vectorizer is not None and user_history is not None and not user_history.empty:
                for product_id, data in recommendations.items():
                    product_info = self.products_df[self.products_df['product_id'] == product_id]
                    if product_info.empty:
                        continue
                    
                    content_boost = 0
                    for _, hist_item in user_history.head(5).iterrows():
                        hist_item_info = self.products_df[self.products_df['product_id'] == hist_item['product_id']]
                        if hist_item_info.empty:
                            continue
                        
                        if 'search_text' in hist_item_info.columns and 'search_text' in product_info.columns:
                            hist_text = hist_item_info['search_text'].iloc[0]
                            prod_text = product_info['search_text'].iloc[0]
                            
                            similarity = SequenceMatcher(None, hist_text, prod_text).ratio()
                            content_boost += similarity * 0.2
                    
                    recommendations[product_id]['score'] += content_boost
            
            results = []
            for product_id, data in sorted(recommendations.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]:
                results.append({
                    'product_id': product_id,
                    'similarity_score': min(1.0, data['score']),
                    'product_name': data['product_name'],
                    'category_id': data['category_id'],
                    'price': data['price'],
                    'seller_id': data['seller_id'],
                    'category_name': CATEGORY_MAP.get(data['category_id'], {}).get('name', 'Unknown')
                })
            
            if len(results) < top_n:
                existing_ids = {r['product_id'] for r in results}
                existing_ids.update(exclude_ids)
                additional_items = self._get_personalized_new_user_recommendations(
                    user_id, top_n - len(results), existing_ids
                )
                results.extend(additional_items)
            
            return results[:top_n]
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering recommendations: {e}")
            return self._get_personalized_new_user_recommendations(user_id, top_n, user_owned_items)
    
    def _get_recommendations_from_user_items(self, user_posted_items, user_id, top_n, user_owned_items):
        """Get recommendations based on items user has posted"""
        recommendations = {}
        
        for _, user_item in user_posted_items.iterrows():
            item_name = user_item['product_name']
            item_id = user_item['product_id']
            
            similar_items = self.semantic_engine.find_similar_by_name(
                item_name, exclude_id=item_id, top_n=5, exclude_seller_id=user_id
            )
            
            for similar_item in similar_items:
                product_id = similar_item['product_id']
                score = similar_item['similarity_score']
                
                if product_id in user_owned_items:
                    continue
                
                if product_id not in recommendations:
                    recommendations[product_id] = 0
                recommendations[product_id] += score
        
        results = []
        for product_id, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                product_info = product_info.iloc[0]
                if product_info['seller_id'] != user_id:
                    results.append({
                        'product_id': product_id,
                        'similarity_score': float(min(1.0, score)),
                        'product_name': product_info['product_name'],
                        'category_id': product_info['category_id'],
                        'price': product_info['price'],
                        'seller_id': product_info['seller_id'],
                        'category_name': CATEGORY_MAP.get(product_info['category_id'], {}).get('name', 'Unknown')
                    })
        
        if len(results) < top_n:
            existing_ids = {r['product_id'] for r in results}
            existing_ids.update(user_owned_items)
            additional_items = self._get_personalized_new_user_recommendations(
                user_id, top_n - len(results), existing_ids
            )
            results.extend(additional_items)
        
        return results[:top_n]
    
    def _get_personalized_new_user_recommendations(self, user_id, top_n, exclude_ids=None):
        """Get personalized recommendations for new users based on the user model."""
        if exclude_ids is None:
            exclude_ids = set()

        logger.info(f"Generating new user recommendations for user {user_id} using model_user.")
        
        user_preferences = None
        if self.model_manager.model_user:
            try:
                user_features = np.array([user_id]) 
                predicted_scores = self.model_manager.predict_user_preferences(user_features)
                
                if predicted_scores is not None and len(predicted_scores) > 0:
                    scores = predicted_scores[0]
                    user_preferences = {i + 1: score for i, score in enumerate(scores)}
                    logger.info(f"Model-based preferences for user {user_id}: {user_preferences}")

            except Exception as e:
                logger.error(f"Failed to get preferences from model_user for user {user_id}: {e}")
                user_preferences = None
        
        if user_preferences is None:
            logger.warning(f"Falling back to hash-based preferences for user {user_id}.")
            user_preferences = self._generate_user_preference_profile(user_id)
        
        user_seed = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        random.seed(user_seed)

        sorted_preferences = sorted(user_preferences.items(), key=lambda item: item[1], reverse=True)

        results = []
        
        for category_id, preference_score in sorted_preferences:
            if len(results) >= top_n:
                break
            
            items_from_category = max(1, int(np.ceil(top_n * preference_score)))
            
            category_products = self.products_df[
                (self.products_df['category_id'] == category_id) &
                (~self.products_df['product_id'].isin(exclude_ids)) &
                (self.products_df['seller_id'] != user_id)
            ]
            
            if not category_products.empty:
                sample_size = min(items_from_category, len(category_products))
                sampled_products = category_products.sample(n=sample_size, random_state=user_seed)
                
                for _, product in sampled_products.iterrows():
                    if len(results) >= top_n: break
                    
                    results.append({
                        'product_id': product['product_id'],
                        'similarity_score': float(preference_score),
                        'product_name': product['product_name'],
                        'category_id': product['category_id'],
                        'price': product['price'],
                        'seller_id': product['seller_id'],
                        'category_name': CATEGORY_MAP.get(product['category_id'], {}).get('name', 'Unknown')
                    })
                    exclude_ids.add(product['product_id'])

        if len(results) < top_n:
            remaining_products = self.products_df[
                (~self.products_df['product_id'].isin(exclude_ids)) &
                (self.products_df['seller_id'] != user_id)
            ]
            
            if len(remaining_products) > 0:
                additional_needed = top_n - len(results)
                sample_size = min(additional_needed, len(remaining_products))
                additional_products = remaining_products.sample(n=sample_size, random_state=user_seed + 1)
                
                for _, product in additional_products.iterrows():
                    results.append({
                        'product_id': product['product_id'],
                        'similarity_score': 0.4 + np.random.uniform(-0.1, 0.1),
                        'product_name': product['product_name'],
                        'category_id': product['category_id'],
                        'price': product['price'],
                        'seller_id': product['seller_id'],
                        'category_name': CATEGORY_MAP.get(product['category_id'], {}).get('name', 'Unknown')
                    })
        
        random.seed()
        return results[:top_n]
    
    def _generate_user_preference_profile(self, user_id):
        """Generate consistent user preference profile based on user_id"""
        user_hash = hashlib.md5(str(user_id).encode()).hexdigest()
        
        pref1 = int(user_hash[0:2], 16) / 255.0
        pref2 = int(user_hash[2:4], 16) / 255.0
        pref3 = int(user_hash[4:6], 16) / 255.0
        
        total = pref1 + pref2 + pref3
        if total > 0:
            pref1 /= total
            pref2 /= total
            pref3 /= total
        else:
            pref1 = pref2 = pref3 = 1/3
        
        return {
            1: pref1,
            2: pref2,
            3: pref3
        }
    
    def get_item_recommendations(self, product_id, user_id, top_n=10):
        """Get similar items using pre-calculated cache, prioritize same category as the clicked product, only available products."""
        try:
            user_owned_items = DatabaseManager.get_user_owned_items(user_id)
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if product_info.empty:
                return self._get_personalized_new_user_recommendations(user_id, top_n, user_owned_items)
            clicked_category = product_info.iloc[0]['category_id']
            # Only available products
            available_ids = set(self.products_df[self.products_df.get('status', 'available').str.lower() == 'available']['product_id'].tolist()) \
                if 'status' in self.products_df.columns else set(self.products_df['product_id'].tolist())

            if product_id in self.item_similarity_cache:
                similar_items = self.item_similarity_cache[product_id][:top_n * 5]
                filtered_items = []
                for item in similar_items:
                    if len(filtered_items) >= top_n:
                        break
                    if item['product_id'] in user_owned_items or item['product_id'] not in available_ids:
                        continue
                    prod_info = self.products_df[self.products_df['product_id'] == item['product_id']]
                    if not prod_info.empty:
                        prod_info = prod_info.iloc[0]
                        if prod_info['seller_id'] == user_id:
                            continue
                        if prod_info['category_id'] == clicked_category:
                            filtered_items.append({
                                'product_id': item['product_id'],
                                'similarity_score': item['similarity_score'],
                                'product_name': prod_info['product_name'],
                                'category_id': prod_info['category_id'],
                                'price': prod_info['price'],
                                'seller_id': prod_info['seller_id'],
                                'category_name': CATEGORY_MAP.get(prod_info['category_id'], {}).get('name', 'Unknown')
                            })
                if len(filtered_items) < top_n:
                    existing_ids = {r['product_id'] for r in filtered_items}
                    existing_ids.update(user_owned_items)
                    additional_items = self._get_category_recommendations(
                        clicked_category, product_id, top_n - len(filtered_items), user_id, existing_ids, available_ids
                    )
                    filtered_items.extend(additional_items)
                return filtered_items[:top_n]
            else:
                return self._get_category_recommendations(clicked_category, product_id, top_n, user_id, user_owned_items, available_ids)
        except Exception as e:
            logger.error(f"Error in item recommendations: {e}")
            return self._get_personalized_new_user_recommendations(user_id, top_n, set())

    def _get_category_recommendations(self, category_id, exclude_product_id, top_n, user_id, exclude_ids=None, available_ids=None):
        """Get recommendations from same category, only available products."""
        if exclude_ids is None:
            exclude_ids = set()
        if available_ids is None:
            available_ids = set(self.products_df[self.products_df.get('status', 'available').str.lower() == 'available']['product_id'].tolist()) \
                if 'status' in self.products_df.columns else set(self.products_df['product_id'].tolist())
        category_products = self.products_df[
            (self.products_df['category_id'] == category_id) &
            (self.products_df['product_id'] != exclude_product_id) &
            (~self.products_df['product_id'].isin(exclude_ids)) &
            (self.products_df['seller_id'] != user_id) &
            (self.products_df['product_id'].isin(available_ids))
        ].head(top_n)
        results = []
        for _, product in category_products.iterrows():
            results.append({
                'product_id': product['product_id'],
                'similarity_score': 0.8,
                'product_name': product['product_name'],
                'category_id': product['category_id'],
                'price': product['price'],
                'seller_id': product['seller_id'],
                'category_name': CATEGORY_MAP.get(product['category_id'], {}).get('name', 'Unknown')
            })
        return results

# Global engine instance
engine = SmartRecommendationEngine()

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10

class SearchRequest(BaseModel):
    keyword: str
    user_id: int
    top_n: int = 10

class ItemRequest(BaseModel):
    product_id: int
    user_id: int
    top_n: int = 10

class ProductRecommendation(BaseModel):
    product_id: int
    product_name: str
    seller_id: int
    product_price: float
    similarity_score: float
    category_id: int
    category_name: str
    recommendation_reason: str

class RecommendationResponse(BaseModel):
    recommendations: List[ProductRecommendation]
    total_found: int
    algorithm_used: str
    query_info: str

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Fast startup"""
    logger.info("Starting recommendation system...")
    engine.initialize()

@app.post("/refresh_data")
async def refresh_data():
    """Refresh data and models"""
    try:
        success = engine.refresh_data()
        return {
            "success": success,
            "message": "Data refreshed successfully" if success else "Refresh failed",
            "timestamp": datetime.now().isoformat(),
            "products_loaded": len(engine.products_df) if engine.products_df is not None else 0,
            "models_loaded": engine.model_manager.models_loaded,
            "last_refresh": engine.model_manager.last_refresh.isoformat() if engine.model_manager.last_refresh else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/user", response_model=RecommendationResponse)
def recommend_user(req: RecommendationRequest):
    logger.info(f"Received /recommend/user request: {req}")
    try:
        if not engine.is_initialized:
            engine.initialize()
        recommendations_data = engine.get_user_recommendations(req.user_id, req.top_n)
        recommendations = []
        for rec in recommendations_data:
            recommendations.append(ProductRecommendation(
                product_id=rec.get('product_id', 0),
                product_name=rec.get('product_name', ''),
                seller_id=rec.get('seller_id', 0),
                product_price=rec.get('price', 0.0),
                similarity_score=rec.get('similarity_score', 0.0),
                category_id=rec.get('category_id', 0),
                category_name=rec.get('category_name', ''),
                recommendation_reason=f"Personalized for user {req.user_id} (score: {rec.get('similarity_score', 0.0):.3f})"
            ))
        logger.info(f"Returning {len(recommendations)} recommendations for user {req.user_id}")
        return RecommendationResponse(
            recommendations=recommendations,
            total_found=len(recommendations),
            algorithm_used="hybrid_model_based_recommendation",
            query_info=f"User {req.user_id} - personalized recommendations (excluding owned items)"
        )
    except Exception as e:
        logger.error(f"Error in user recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/search", response_model=RecommendationResponse)
def recommend_search(req: SearchRequest):
    logger.info(f"Received /recommend/search request: {req}")
    try:
        if not engine.is_initialized:
            engine.initialize()
        # --- FIX: panggil method semantic_engine.search, bukan engine.search_recommendations ---
        semantic_results = engine.semantic_engine.search(req.keyword, 80, exclude_seller_id=req.user_id)
        query_lower = req.keyword.lower()
        category_scores = {}
        for cat_id, cat_info in CATEGORY_MAP.items():
            score = sum(1 for kw in cat_info['keywords'] if kw in query_lower)
            if score > 0:
                category_scores[cat_id] = score
        target_category = max(category_scores, key=category_scores.get) if category_scores else None

        hasil = []
        for rec in semantic_results:
            # --- FILTER: hanya tampilkan produk dengan kategori yang sama dengan target_category ---
            if target_category and rec.get('category_id', 0) != target_category:
                continue
            idx = None
            if hasattr(engine, "products_df"):
                df = engine.products_df
                idxs = df.index[df['product_id'] == rec['product_id']].tolist()
                if idxs:
                    idx = idxs[0]
            if idx is None:
                continue

            if hasattr(engine.semantic_engine, "tfidf_matrix"):
                produk_tfidf_vec = engine.semantic_engine.tfidf_matrix[idx].toarray().flatten()
            else:
                produk_tfidf_vec = np.zeros(500)

            category_id = rec.get('category_id', 0)
            NUM_CATEGORIES = 10
            category_one_hot = np.zeros(NUM_CATEGORIES)
            if 0 < category_id <= NUM_CATEGORIES:
                category_one_hot[category_id-1] = 1

            expected_input_dim = model_search.input_shape[-1]
            tfidf_len = expected_input_dim - NUM_CATEGORIES
            if len(produk_tfidf_vec) > tfidf_len:
                produk_tfidf_vec = produk_tfidf_vec[:tfidf_len]
            elif len(produk_tfidf_vec) < tfidf_len:
                produk_tfidf_vec = np.pad(produk_tfidf_vec, (0, tfidf_len - len(produk_tfidf_vec)), mode='constant')
            
            x_input = np.concatenate([category_one_hot, produk_tfidf_vec]).reshape(1, -1)
            if x_input.shape[1] != expected_input_dim:
                continue
            prob = float(model_search.predict(x_input, verbose=0)[0][0])

            hasil.append((prob, rec))

        hasil_sorted = sorted(hasil, key=lambda x: x[0], reverse=True)
        top_n = req.top_n
        top_k = min(2, top_n)
        diverse_k = top_n - top_k
        top_recs = hasil_sorted[:top_k]
        diverse_pool = hasil_sorted[top_k:]
        if diverse_pool and diverse_k > 0:
            diverse_recs = random.sample(diverse_pool, min(diverse_k, len(diverse_pool)))
        else:
            diverse_recs = []
        final_recs = top_recs + diverse_recs
        random.shuffle(final_recs)

        recommendations = []
        for score, rec in final_recs:
            if score > 0.7:
                relevance = "High match"
            elif score > 0.4:
                relevance = "Good match"
            else:
                relevance = "Related item"
            recommendations.append(ProductRecommendation(
                product_id=rec.get('product_id', 0),
                product_name=rec.get('product_name', ''),
                seller_id=rec.get('seller_id', 0),
                product_price=rec.get('price', 0.0),
                similarity_score=score,
                category_id=rec.get('category_id', 0),
                category_name=rec.get('category_name', ''),
                recommendation_reason=f"{relevance} for '{req.keyword}' (score: {score:.3f})"
            ))
        logger.info(f"Returning {len(recommendations)} strict-category search recommendations for keyword '{req.keyword}'")
        return RecommendationResponse(
            recommendations=recommendations,
            total_found=len(recommendations),
            algorithm_used="semantic_search+model_search+strict_category",
            query_info=f"Search: '{req.keyword}' - only exact category match"
        )
    except Exception as e:
        logger.error(f"Error in search recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/item", response_model=RecommendationResponse)
def recommend_item(req: ItemRequest):
    logger.info(f"Received /recommend/item request: {req}")
    try:
        if not engine.is_initialized:
            engine.initialize()
        recommendations_data = engine.get_item_recommendations(req.product_id, req.user_id, req.top_n)
        recommendations = []
        for rec in recommendations_data:
            if rec.get('similarity_score', 0.0) > 0.8:
                similarity_type = "Very similar"
            elif rec.get('similarity_score', 0.0) > 0.6:
                similarity_type = "Similar"
            else:
                similarity_type = "Related"
            recommendations.append(ProductRecommendation(
                product_id=rec.get('product_id', 0),
                product_name=rec.get('product_name', ''),
                seller_id=rec.get('seller_id', 0),
                product_price=rec.get('price', 0.0),
                similarity_score=rec.get('similarity_score', 0.0),
                category_id=rec.get('category_id', 0),
                category_name=rec.get('category_name', ''),
                recommendation_reason=f"{similarity_type} to product {req.product_id} (score: {rec.get('similarity_score', 0.0):.3f})"
            ))
        logger.info(f"Returning {len(recommendations)} item recommendations for product {req.product_id}")
        return RecommendationResponse(
            recommendations=recommendations,
            total_found=len(recommendations),
            algorithm_used="model_item_embedding_similarity",
            query_info=f"Similar to product {req.product_id} (excluding user {req.user_id} owned items)"
        )
    except Exception as e:
        logger.error(f"Error in item recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check with performance info"""
    return {
        "status": "healthy",
        "engine_initialized": engine.is_initialized,
        "models_loaded": engine.model_manager.models_loaded,
        "last_update": engine.last_update.isoformat() if engine.last_update else None,
        "products_count": len(engine.products_df) if engine.products_df is not None else 0,
        "similarity_cache_size": len(engine.item_similarity_cache),
        "categories": list(CATEGORY_MAP.keys())
    }

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Smart Recommendation System v6.0 - Fully Model-Integrated",
        "features": [
            "✅ Personalized recommendations for each user (different results per user)",
            "✅ Item-to-item recommendations using semantic embeddings from `model_item`",
            "✅ Cold-start user recommendations using preference prediction from `model_user`",
            "✅ Intelligent search using TF-IDF retrieval and `model_search` re-ranking",
            "✅ Excludes user's own items from all recommendations"
        ],
        "user_recommendation_logic": {
            "existing_users": "Based on collaborative filtering or similarity to items they've posted",
            "new_users": "Personalized based on `model_user` predictions (data-driven)",
            "validation": "All recommendations exclude items owned by the requesting user"
        },
        "item_recommendation_logic": {
            "primary_method": "Finds nearest neighbors in the embedding space generated by `model_item`",
            "efficiency": "Uses a pre-calculated cache for fast lookups",
            "fallback": "Recommends items from the same category if primary fails"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)