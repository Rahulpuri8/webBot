




from fastapi import APIRouter, Request
import asyncio
from src.db import db
from src.product_handler import product_search
from src.watch_rag import faq_search, llm
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

chat_router = APIRouter()

# Convert to proper ChatPromptTemplate
RESPONSE_EVAL_PROMPT = ChatPromptTemplate.from_template("""
You are a quality control specialist for a watch store chatbot. Evaluate if this response fully answers the query.

Original Query: {query}
Response Type: {response_type}
Response Content: {response}

Check for:
1. Completeness - Does it answer all parts of the query?
2. Accuracy - Is the information correct?
3. Clarity - Is it easy to understand?

Provide your enhanced response. If the existing response is good, you may keep it as-is. If not, improve it.

Final Answer:""")

async def enhance_response(query: str, response: str, response_type: str) -> str:
    """Use LLM to validate and enhance all responses"""
    try:
        # Create a proper prompt value
        prompt_value = RESPONSE_EVAL_PROMPT.format_messages(
            query=query,
            response_type=response_type,
            response=response
        )
        
        # Invoke the LLM with the proper prompt
        result = await llm.ainvoke(prompt_value)
        return str(result.content) if hasattr(result, 'content') else str(result)
    except Exception as e:
        print(f"LLM enhancement error: {str(e)}")
        return response  # Return original response if enhancement fails

@chat_router.post("/chat")
async def chat(req: Request):
    try:
        data = await req.json()
        message = data.get("message", "").strip()
        if not message:
            return {"error": "Empty query"}
        
        # Process queries in parallel
        product_task = product_search.handle_query(message)
        faq_task = faq_search.handle_query(message)
        product_answer, faq_answer = await asyncio.gather(product_task, faq_task)
        
        # Determine response type
        if product_answer.startswith("No") and faq_answer.startswith("No"):
            intent = "unknown"
            response = await enhance_response(message, "I couldn't find information about that.", "fallback")
        elif product_answer.startswith("No"):
            intent = "faq"
            response = await enhance_response(message, faq_answer, "faq")
        elif faq_answer.startswith("No"):
            intent = "product"
            response = await enhance_response(message, product_answer, "product")
        else:
            intent = "combined"
            combined = f"Product Info: {product_answer}\nFAQ Info: {faq_answer}"
            response = await enhance_response(message, combined, "combined")
        
        return {"intent": intent, "response": response}
    
    except Exception as e:
        print(f"Route error: {str(e)}")
        return {"error": "Processing failed"}















# from fastapi import APIRouter, Request
# import asyncio
# from src.db import db
# from src.product_handler import product_search
# from src.watch_rag import faq_search, llm
# import time

# chat_router = APIRouter()
# conversations_collection = db.conversations

# async def process_query(message: str):
#     start_time = time.time()
    
#     # Parallel execution
#     product_task = asyncio.to_thread(product_search.handle_query, message)
#     faq_task = asyncio.to_thread(faq_search.handle_query, message)
#     product_answer, faq_answer = await asyncio.gather(product_task, faq_task)
    
#     # Response logic
#     if product_answer.startswith("No") and faq_answer.startswith("No"):
#         intent = "unknown"
#         response = "I couldn't find information about that."
#     elif product_answer.startswith("No"):
#         intent = "faq"
#         response = faq_answer
#     elif faq_answer.startswith("No"):
#         intent = "product"
#         response = product_answer
#     else:
#         intent = "combined"
#         response = f"Product: {product_answer}\nFAQ: {faq_answer}"
    
#     print(f"Total route execution: {time.time()-start_time:.4f}s")
#     return intent, response

# @chat_router.post("/chat")
# async def chat(req: Request):
#     try:
#         data = await req.json()
#         message = data.get("message", "").strip()
#         if not message:
#             return {"error": "Empty query"}
        
#         intent, response = await process_query(message)
#         return {"intent": intent, "response": response}
    
#     except Exception as e:
#         print(f"Route error: {str(e)}")
#         return {"error": "Processing failed"}






















# # ---------------- src/routes.py ----------------
# from fastapi import APIRouter, Request
# from src.product_handler import handle_product_query
# from src.watch_rag import handle_faq_query, llm_chain
# from src.db import db
# import asyncio
# import time



# chat_router = APIRouter()
# conversations_collection = db.conversations

# @chat_router.post("/chat")
# async def chat(req: Request):
#     start_time = time.time()

#     data = await req.json()
#     message = data.get("message")

#     product_answer = handle_product_query(message)
#     faq_answer = handle_faq_query(message)

#     # Determine intent
#     has_product_info = "Sorry" not in product_answer
#     has_faq_info = "Sorry" not in faq_answer

#     if has_product_info and has_faq_info:
#         intent = "product_info + faq"
#     elif has_product_info:
#         intent = "product_info"
#     elif has_faq_info:
#         intent = "faq"
#     else:
#         intent = "unknown"

#     # Generate clean response
#     if intent == "unknown":
#         response_text = "I'm not sure how to help with that. Try asking about our watches or store policies."
#     else:
#         result = llm_chain.invoke({
#             "question": message,
#             "product_info": product_answer if has_product_info else "No product information available",
#             "faq_info": faq_answer if has_faq_info else "No FAQ information available"
#         })
#         response_text = result["text"] if isinstance(result, dict) else str(result)

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Execution time (Routes): {execution_time:.4f} seconds")

#     return {
#         "intent": intent,
#         "response": response_text  # Return just the clean text
#     }




















# # ---------------- src/routes.py ----------------
# from fastapi import APIRouter, Request
# from src.product_handler import handle_product_query
# from src.watch_rag import handle_faq_query, llm_chain
# from src.db import db

# chat_router = APIRouter()
# conversations_collection = db.conversations

# @chat_router.post("/chat")
# async def chat(req: Request):
#     data = await req.json()
#     message = data.get("message")

#     product_answer = handle_product_query(message)
#     faq_answer = handle_faq_query(message)

#     # Determine intent
#     has_product_info = "Sorry" not in product_answer
#     has_faq_info = "Sorry" not in faq_answer

#     if has_product_info and has_faq_info:
#         intent = "product_info + faq"
#     elif has_product_info:
#         intent = "product_info"
#     elif has_faq_info:
#         intent = "faq"
#     else:
#         intent = "unknown"

#     # Generate clean response
#     if intent == "unknown":
#         response_text = "I'm not sure how to help with that. Try asking about our watches or store policies."
#     else:
#         result = llm_chain.invoke({
#             "question": message,
#             "product_info": product_answer if has_product_info else "No product information available",
#             "faq_info": faq_answer if has_faq_info else "No FAQ information available"
#         })
#         response_text = result["text"] if isinstance(result, dict) else str(result)

#     return {
#         "intent": intent,
#         "response": response_text  # Return just the clean text
#     }



















# # ---------------- src/routes.py ----------------
# from fastapi import APIRouter, Request
# from src.intent_classifier import classify_intent
# from src.product_handler import handle_product_query
# from src.watch_rag import handle_faq_query
# from src.db import db
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_community.llms import Ollama

# chat_router = APIRouter()
# conversations_collection = db["conversations"]

# @chat_router.post("/chat")
# async def chat(req: Request):
#     data = await req.json()
#     message = data.get("message")

#     # Determine if both product and FAQ intents exist
#     product_answer = handle_product_query(message)
#     faq_answer = handle_faq_query(message)

#     # Check if either gave relevant responses
#     has_product_info = "Sorry" not in product_answer
#     has_faq_info = "Sorry" not in faq_answer


#     llm = Ollama(model="mistral:latest")

#     # New reasoning prompt
#     reasoning_prompt = PromptTemplate(
#         input_variables=["question", "product_info", "faq_info"],
#         template="""
#             You are a helpful assistant for a watch store.

#             Answer the customer's question using the info provided below. Only use relevant details. If something is not mentioned, do not make it up.

#             Question: {question}

#             Product Info:
#             {product_info}

#             FAQ Info:
#             {faq_info}

#             Answer:
#             """)

#     llm_chain = LLMChain(prompt=reasoning_prompt, llm=llm)

    
#     if has_product_info or has_faq_info:
#         intent = "product_info + faq" if has_product_info and has_faq_info else ("product_info" if has_product_info else "faq")
#         answer = llm_chain.invoke({
#             "question": message,
#             "product_info": product_answer if has_product_info else "None",
#             "faq_info": faq_answer if has_faq_info else "None"
#         })
#     elif has_product_info:
#         intent = "product_info"
#         answer = product_answer
#     elif has_faq_info:
#         intent = "faq"
#         answer = faq_answer
#     else:
#         intent = "unknown"
#         answer = "I'm not sure how to help with that. Try asking about our watches or store policies."

#     conversations_collection.insert_one({
#         "user": message,
#         "bot": str(answer),  # Ensure it's string
#         "intent": intent
#     })


#     return {"intent": intent, "response": answer}







# # ---------------- src/routes.py ----------------
# from fastapi import APIRouter, Request
# from src.intent_classifier import classify_intent
# from src.product_handler import handle_product_query
# from src.watch_rag import handle_faq_query

# chat_router = APIRouter()

# @chat_router.post("/chat")
# async def chat(req: Request):
#     data = await req.json()
#     message = data.get("message")

#     intent = classify_intent(message)

#     if intent == "product_info":
#         answer = handle_product_query(message)
#     elif intent == "faq":
#         answer = handle_faq_query(message)
#     else:
#         answer = "I'm not sure how to help with that. Try asking about our watches or store policies."

#     return {"intent": intent, "response": answer}
