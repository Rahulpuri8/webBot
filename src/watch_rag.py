


import json
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from functools import lru_cache
import time
import asyncio

llm = OllamaLLM(model="mistral:latest")

class FAQSearch:
    def __init__(self):
        self.index = None
        self.documents = []
        self._initialize()

    def _initialize(self):
        """Initialize with pre-computed embeddings"""
        start_time = time.time()
        
        with open("rag_source/faq.json") as f:
            faqs = json.load(f)
        
        self.documents = [Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}") 
                         for faq in faqs]
        
        # Generate embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        texts = [doc.page_content for doc in self.documents]
        vectors = np.array(embeddings.embed_documents(texts)).astype('float32')
        
        # Create appropriate index based on dataset size
        dimension = vectors.shape[1]
        if len(vectors) < 100:  # Small dataset
            self.index = faiss.IndexFlatL2(dimension)
        else:  # Large dataset
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, min(len(vectors)//4, 256))
            self.index.train(vectors)
        
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        print(f"FAQ index initialized in {time.time()-start_time:.2f}s")

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings.embed_query(text)

    async def _format_with_llm(self, query: str, answer: str) -> str:
        """Use LLM to refine FAQ answers"""
        prompt = f"""As a customer service expert, improve this FAQ answer for the specific query.

            Original Query: {query}
            FAQ Answer: {answer}

            Make sure to:
            1. Directly address the query
            2. Keep it concise but complete
            3. Maintain a helpful tone

            Improved Response:"""
        
        try:
            result = await llm.ainvoke(prompt)
            return str(result).strip()
        except Exception as e:
            print(f"LLM formatting error: {str(e)}")
            return answer

    async def handle_query(self, query: str) -> str:
        start_time = time.time()
        try:
            query_embedding = self._get_cached_embedding(query)
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            k = min(3, len(self.documents))
            distances, indices = self.index.search(query_embedding, k)
            
            if len(indices) > 0 and indices[0][0] >= 0:
                doc = self.documents[indices[0][0]]
                answer = doc.page_content.split("\nA: ")[1]
                response = await self._format_with_llm(query, answer)
                print(f"FAQ search took {time.time()-start_time:.4f}s")
                return response
            
            return await self._format_with_llm(query, "No relevant FAQ found.")
        except Exception as e:
            print(f"FAQ search error: {str(e)}")
            return "Error processing FAQ query"

# Global instance initialized at startup
faq_search = FAQSearch()










# import json
# import numpy as np
# import faiss
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from functools import lru_cache
# import time

# class FAQSearch:
#     def __init__(self):
#         self.index = None
#         self.documents = []
#         self._initialize()

#     def _initialize(self):
#         """Initialize with pre-computed embeddings and optimized index"""
#         start_time = time.time()
        
#         with open("rag_source/faq.json") as f:
#             faqs = json.load(f)
        
#         self.documents = [Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}") 
#                          for faq in faqs]
        
#         # Generate embeddings
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         texts = [doc.page_content for doc in self.documents]
#         vectors = np.array(embeddings.embed_documents(texts)).astype('float32')
        
#         # Choose appropriate index type based on dataset size
#         dimension = vectors.shape[1]
#         num_vectors = len(vectors)
        
#         if num_vectors < 100:  # Small dataset - use flat index
#             print(f"Using FlatL2 index for {num_vectors} vectors")
#             self.index = faiss.IndexFlatL2(dimension)
#             faiss.normalize_L2(vectors)
#             self.index.add(vectors)
#         else:  # Larger dataset - use IVF index
#             nlist = min(num_vectors // 4, 256)  # Number of clusters
#             print(f"Using IVF index with {nlist} clusters for {num_vectors} vectors")
#             quantizer = faiss.IndexFlatL2(dimension)
#             self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
#             faiss.normalize_L2(vectors)
#             self.index.train(vectors)
#             self.index.add(vectors)
#             self.index.nprobe = min(10, nlist)  # Number of clusters to explore
        
#         print(f"FAQ index initialized in {time.time()-start_time:.2f}s")

#     @lru_cache(maxsize=1000)
#     def _get_cached_embedding(self, text: str):
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         return embeddings.embed_query(text)

#     def handle_query(self, query: str) -> str:
#         start_time = time.time()
#         try:
#             query_embedding = self._get_cached_embedding(query)
#             query_embedding = np.array([query_embedding]).astype('float32')
#             faiss.normalize_L2(query_embedding)
            
#             k = min(3, len(self.documents))  # Don't request more results than available
#             distances, indices = self.index.search(query_embedding, k)
            
#             if len(indices) > 0 and indices[0][0] >= 0:  # Valid result
#                 doc = self.documents[indices[0][0]]
#                 answer = doc.page_content.split("\nA: ")[1]
#                 print(f"FAQ search took {time.time()-start_time:.4f}s")
#                 return answer
            
#             return "No relevant FAQ found."
#         except Exception as e:
#             print(f"FAQ search error: {str(e)}")
#             return "Error processing FAQ query"

# # Global instance initialized at startup
# faq_search = FAQSearch()
# llm = OllamaLLM(model="mistral:latest")





















# import json
# import numpy as np
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from langchain_core.prompts import PromptTemplate
# import time

# # Load FAQs
# with open("rag_source/faq.json") as f:
#     faqs = json.load(f)

# # Create documents
# documents = [
#     Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}") for faq in faqs
# ]

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": False},
# )

# # Create FAISS vectorstore - let LangChain handle the index creation
# vectorstore = FAISS.from_documents(documents, embeddings)

# # Initialize LLM and QA chain
# llm = OllamaLLM(model="mistral:latest")
# retriever = vectorstore.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, retriever=retriever, return_source_documents=False
# )


# def handle_faq_query(query: str) -> str:
#     start_time = time.time()
#     try:
#         result = qa_chain.invoke(query)
#         if isinstance(result, dict):
#             result = result.get("result", "")
#         if not result or "not provided" in result.lower():
#             return "Sorry, no FAQ information found."

#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Execution time (Rag Query): {execution_time:.4f} seconds")

#         return result
#     except Exception as e:
#         print(f"Error in FAQ query processing: {str(e)}")
#         return "Sorry, I encountered an error processing your request."


# prompt = PromptTemplate.from_template(
#     """
#     You are a watch expert chatbot. Answer clearly and concisely using ONLY the information provided below.
#     If the information is not available, say you don't know.
    
#     Question: {question}

#     {product_info}
    
#     {faq_info}

#     Answer in one short paragraph:
#     """
# )

# llm_chain = prompt | llm




















# # ---------------- src/watch_rag.py ----------------
# import json
# from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# import faiss
# import numpy as np


# # from langchain_community.llms import Ollama
# from langchain_ollama import OllamaLLM

# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# # from langchain.chains import LLMChain

# from langchain_core.runnables import RunnableSequence  # or just use the pipe syntax
# from langchain_core.prompts import PromptTemplate

# with open("rag_source/faq.json") as f:
#     faqs = json.load(f)

# documents = [Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}") for faq in faqs]

# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
#                                    model_kwargs=model_kwargs,
#                                    encode_kwargs=encode_kwargs)


# # # Current (no index optimization)
# # vectorstore = FAISS.from_documents(documents, embeddings)

# # # Faster:
# # vectorstore = FAISS.from_documents(documents, embeddings)
# # vectorstore.index = faiss.IndexHNSWFlat(384, 32)  # 32 = number of neighbors
# # vectorstore.index.add(embeddings)


# # Generate embeddings for all documents
# document_texts = [doc.page_content for doc in documents]
# document_embeddings = embeddings.embed_documents(document_texts)
# document_embeddings = np.array(document_embeddings).astype('float32')

# # Create optimized FAISS index
# dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
# hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
# hnsw_index.add(document_embeddings)


# # Create proper document store and index mapping
# docstore = {str(i): doc for i, doc in enumerate(documents)}
# index_to_docstore_id = {i: str(i) for i in range(len(documents))}


# # Create FAISS vectorstore with custom index
# vectorstore = FAISS(
#     embeddings.embed_query,
#     index=hnsw_index,
#     docstore=documents,
#     index_to_docstore_id=index_to_docstore_id
#     )


# llm = OllamaLLM(model="mistral:latest")
# retriever = vectorstore.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)


# # # In routes.py, modify the response handling:
# # def clean_response(text):
# #     return ' '.join(text.split()).replace('\n', ' ').strip()


# # def handle_faq_query(query: str) -> str:
# #     result = qa_chain.invoke(query)
# #     # result = clean_response(result["text"] if isinstance(result, dict) else str(result))
# #     if isinstance(result, dict):
# #         result = result.get("result", "")
# #     if not result or "not provided" in result:
# #         return "Sorry, no FAQ information found."
# #     return result


# def handle_faq_query(query: str) -> str:
#     try:
#         result = qa_chain.invoke(query)
#         if isinstance(result, dict):
#             result = result.get("result", "")
#         if not result or "not provided" in result:
#             return "Sorry, no FAQ information found."
#         return result
#     except Exception as e:
#         print(f"Error in FAQ query: {str(e)}")
#         return "Sorry, I encountered an error processing your request."


# prompt = PromptTemplate.from_template(
#     """
#     You are a watch expert chatbot. Answer clearly and concisely using ONLY the information provided below.
#     If the information is not available, say you don't know.

#     Question: {question}

#     {product_info}

#     {faq_info}

#     Answer in one short paragraph:
#     """
# )


# # llm_chain = LLMChain(llm=llm, prompt=prompt)

# # If `prompt` is a PromptTemplate and `llm` is an LLM (e.g., OpenAI model):
# llm_chain = prompt | llm
# # response = chain.invoke({"input": "your question"})


# # # Print the answer
# # print(response.content)  # `.content` for Chat models like Ollama


# # It works with the following code

# # ---------------- src/watch_rag.py ----------------
# import json
# from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# import faiss
# import numpy as np


# # from langchain_community.llms import Ollama
# from langchain_ollama import OllamaLLM

# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# # from langchain.chains import LLMChain

# from langchain_core.runnables import RunnableSequence  # or just use the pipe syntax
# from langchain_core.prompts import PromptTemplate

# with open("rag_source/faq.json") as f:
#     faqs = json.load(f)

# documents = [Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}") for faq in faqs]

# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
#                                    model_kwargs=model_kwargs,
#                                    encode_kwargs=encode_kwargs)


# # # Current (no index optimization)
# # vectorstore = FAISS.from_documents(documents, embeddings)

# # # Faster:
# # vectorstore = FAISS.from_documents(documents, embeddings)
# # vectorstore.index = faiss.IndexHNSWFlat(384, 32)  # 32 = number of neighbors
# # vectorstore.index.add(embeddings)


# # Generate embeddings for all documents
# document_texts = [doc.page_content for doc in documents]
# document_embeddings = embeddings.embed_documents(document_texts)
# document_embeddings = np.array(document_embeddings).astype('float32')

# # Create optimized FAISS index
# dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
# hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
# hnsw_index.add(document_embeddings)

# # Create FAISS vectorstore with custom index
# vectorstore = FAISS(embeddings.embed_query, hnsw_index, documents, {})


# llm = OllamaLLM(model="mistral:latest")
# retriever = vectorstore.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)


# # # In routes.py, modify the response handling:
# # def clean_response(text):
# #     return ' '.join(text.split()).replace('\n', ' ').strip()


# # def handle_faq_query(query: str) -> str:
# #     result = qa_chain.invoke(query)
# #     # result = clean_response(result["text"] if isinstance(result, dict) else str(result))
# #     if isinstance(result, dict):
# #         result = result.get("result", "")
# #     if not result or "not provided" in result:
# #         return "Sorry, no FAQ information found."
# #     return result


# def handle_faq_query(query: str) -> str:
#     try:
#         result = qa_chain.invoke(query)
#         if isinstance(result, dict):
#             result = result.get("result", "")
#         if not result or "not provided" in result:
#             return "Sorry, no FAQ information found."
#         return result
#     except Exception as e:
#         print(f"Error in FAQ query: {str(e)}")
#         return "Sorry, I encountered an error processing your request."


# prompt = PromptTemplate.from_template(
#     """
#     You are a watch expert chatbot. Answer clearly and concisely using ONLY the information provided below.
#     If the information is not available, say you don't know.

#     Question: {question}

#     {product_info}

#     {faq_info}

#     Answer in one short paragraph:
#     """
# )


# # llm_chain = LLMChain(llm=llm, prompt=prompt)

# # If `prompt` is a PromptTemplate and `llm` is an LLM (e.g., OpenAI model):
# llm_chain = prompt | llm
# # response = chain.invoke({"input": "your question"})


# # # Print the answer
# # print(response.content)  # `.content` for Chat models like Ollama


# # ---------------- src/watch_rag.py ----------------
# import json
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.schema import Document

# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains.retrieval_qa.base import RetrievalQA


# with open("rag_source/faq.json") as f:
#     faqs = json.load(f)

# documents = [Document(page_content=f"Q: {faq['question']}\nA: {faq['answer']}" ) for faq in faqs]

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(documents, embeddings)

# llm = Ollama(model="mistral:latest")


# custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#         Use only the following context to answer the question. If the answer isn't clearly stated, reply with "Sorry, no FAQ information found."

#         Context:
#         {context}

#         Question: {question}
#         Answer:"""
#     )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type="stuff",
#     return_source_documents=False,
#     chain_type_kwargs={"prompt": custom_prompt}
# )


# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=False)

# def handle_faq_query(query: str) -> str:
#     result = qa_chain.invoke(query)
#     if isinstance(result, dict):
#         result = result.get("text", "") or result.get("faq_info", "")
#     if not isinstance(result, str) or not result.strip():
#         return "Sorry, no FAQ information found."
#     return result.strip()
