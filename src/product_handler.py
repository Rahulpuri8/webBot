



from src.db import products_collection
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process
import numpy as np
import faiss
from functools import lru_cache
import time
from langchain_ollama import OllamaLLM
import asyncio

model = SentenceTransformer("all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral:latest")

class ProductSearch:
    def __init__(self):
        self.index = None
        self.product_docs = []
        self._initialize()

    def _initialize(self):
        """Initialize with pre-computed embeddings and optimized FAISS index"""
        start_time = time.time()
        
        # Load only necessary fields
        self.product_docs = list(products_collection.find(
            {}, 
            {"name": 1, "price": 1, "features": 1}
        ))
        
        # Generate embeddings
        product_texts = [f"{doc['name']} {' '.join(doc['features'])}" 
                        for doc in self.product_docs]
        embeddings = model.encode(product_texts)
        
        # Create optimized FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Product index initialized in {time.time()-start_time:.2f}s")

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str):
        return model.encode(text)

    async def _format_with_llm(self, query: str, product_data: dict) -> str:
        """Use LLM to format product responses"""
        prompt = f"""You're a watch expert assistant. Format this product information to answer the query.

            Query: {query}
            Product: {product_data['name']}
            Price: ${product_data['price']}
            Features: {', '.join(product_data['features'])}

            Create a concise, helpful response that:
            1. Directly answers the query
            2. Highlights key features
            3. Uses natural, friendly language

            Response:"""
        
        try:
            result = await llm.ainvoke(prompt)
            return str(result).strip()
        except Exception as e:
            print(f"LLM formatting error: {str(e)}")
            return f"{product_data['name']} costs ${product_data['price']}. Features: {', '.join(product_data['features'])}"

    async def handle_query(self, query: str) -> str:
        start_time = time.time()
        try:
            # Semantic search
            query_embedding = self._get_cached_embedding(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.index.search(query_embedding, 3)
            if distances[0][0] < 0.7:  # Good match threshold
                doc = self.product_docs[indices[0][0]]
                response = await self._format_with_llm(query, doc)
                print(f"Product search took {time.time()-start_time:.4f}s")
                return await response
            
            # Fallback to fuzzy matching
            names = [doc['name'] for doc in self.product_docs]
            match, score, idx = process.extractOne(query, names)
            if score > 75:
                doc = self.product_docs[idx]
                response = await self._format_with_llm(query, doc)
                return response
            
            return await self._format_with_llm(query, {
                "name": "No matching product",
                "price": 0,
                "features": []
            })
        except Exception as e:
            print(f"Product search error: {str(e)}")
            return "Error processing product query"

# Global instance initialized at startup
product_search = ProductSearch()











# from src.db import products_collection
# from sentence_transformers import SentenceTransformer, util
# from rapidfuzz import process
# import numpy as np
# import faiss
# from functools import lru_cache
# import time

# model = SentenceTransformer("all-MiniLM-L6-v2")

# class ProductSearch:
#     def __init__(self):
#         self.index = None
#         self.product_docs = []
#         self._initialize()

#     def _initialize(self):
#         """Initialize with pre-computed embeddings and optimized FAISS index"""
#         start_time = time.time()
        
#         # Load only necessary fields
#         self.product_docs = list(products_collection.find(
#             {}, 
#             {"name": 1, "price": 1, "features": 1}
#         ))
        
#         # Generate embeddings
#         product_texts = [f"{doc['name']} {' '.join(doc['features'])}" 
#                         for doc in self.product_docs]
#         embeddings = model.encode(product_texts)
        
#         # Create optimized FAISS index
#         dimension = embeddings.shape[1]
#         self.index = faiss.IndexHNSWFlat(dimension, 32)
#         self.index.hnsw.efConstruction = 40
#         self.index.hnsw.efSearch = 16
        
#         # Normalize for cosine similarity
#         faiss.normalize_L2(embeddings)
#         self.index.add(embeddings)
        
#         print(f"Product index initialized in {time.time()-start_time:.2f}s")

#     @lru_cache(maxsize=1000)
#     def _get_cached_embedding(self, text: str):
#         return model.encode(text)

#     def handle_query(self, query: str) -> str:
#         start_time = time.time()
#         try:
#             # Semantic search
#             query_embedding = self._get_cached_embedding(query)
#             query_embedding = np.array([query_embedding]).astype('float32')
#             faiss.normalize_L2(query_embedding)
            
#             distances, indices = self.index.search(query_embedding, 3)
#             if distances[0][0] < 0.7:  # Good match threshold
#                 doc = self.product_docs[indices[0][0]]
#                 response = f"{doc['name']} (${doc['price']})"
#                 print(f"Product search took {time.time()-start_time:.4f}s")
#                 return response
            
#             # Fallback to fuzzy matching
#             names = [doc['name'] for doc in self.product_docs]
#             match, score, idx = process.extractOne(query, names)
#             if score > 75:
#                 doc = self.product_docs[idx]
#                 return f"Fuzzy match: {doc['name']} (${doc['price']})"
            
#             return "No matching products found."
#         except Exception as e:
#             print(f"Product search error: {str(e)}")
#             return "Error processing product query"

# # Global instance initialized at startup
# product_search = ProductSearch()
















# # ---------------- src/product_handler.py ----------------
# from src.db import products_collection
# from sentence_transformers import SentenceTransformer, util
# from rapidfuzz import process
# import time
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # # # Current (loads all fields)
# # product_docs = list(products_collection.find({}))

# # Faster (project only needed fields):
# product_docs = list(products_collection.find({}, {"name": 1, "price": 1, "features": 1}))


# product_texts = [f"{doc['name']} {' '.join(doc['features'])}" for doc in product_docs]
# product_embeddings = model.encode(product_texts, convert_to_tensor=True)



# def handle_product_query(message: str):
#     start_time = time.time()

#     query_embedding = model.encode(message, convert_to_tensor=True)
#     similarities = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]

#     # Get best match only
#     best_match_idx = similarities.argmax().item()
#     best_score = similarities[best_match_idx].item()
    
#     if best_score > 0.7:  # Higher threshold for single match
#         doc = product_docs[best_match_idx]
#         return f"{doc['name']} costs ${doc['price']} and includes features like {', '.join(doc['features'])}."
    
#     # Fallback to fuzzy matching
#     names = [doc['name'] for doc in product_docs]
#     match, score, idx = process.extractOne(message, names)
#     if score > 75:  
#         doc = product_docs[idx]
#         return f"{doc['name']} costs ${doc['price']} and includes features like {', '.join(doc['features'])}."

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Execution time (Product Query): {execution_time:.4f} seconds")   

#     return "Sorry, no product information found."


