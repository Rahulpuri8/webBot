





# ---------------- streamlit_app.py ----------------
import streamlit as st
import requests
import pymongo

DEBUG = False  # Set to True to show detailed error tracebacks

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["watch_store"]
conversations = db.conversations


def clean_response(text):
    """Remove extra whitespace and fix formatting"""
    if isinstance(text, dict):
        text = str(text)
    return ' '.join(str(text).split()).replace('\n', ' ').strip()


st.title("⌚ WatchBot - Ask about products or FAQs")

st.subheader("🕓 Conversation History")
history = list(conversations.find().sort("_id", -1).limit(10))
# For the history display, add cleaning:
for convo in reversed(history):
    cleaned_bot = clean_response(convo.get('bot', ''))
    st.markdown(f"**You:** {convo.get('user', '')}")
    st.markdown(f"**Bot:** {cleaned_bot}")

st.subheader("🗨️ Ask a Question")
user_input = st.text_input("You:", placeholder="Ask me about a watch or policy...")

if st.button("Submit") and user_input:
    try:
        response = requests.post(
            "http://localhost:8001/chat",
            json={"message": user_input},
            # timeout=10  # Add timeout
        )
        response.raise_for_status()  # Raise HTTP errors
        data = response.json()
        
        cleaned_response = clean_response(data['response'])
        
        st.markdown(f"**Intent:** {data.get('intent', 'unknown')}")
        st.markdown(f"**Bot:** {cleaned_response}")
        
        conversations.insert_one({
            "user": user_input,
            "bot": cleaned_response,
            "intent": data.get('intent', 'unknown')
        })
        
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
    except ValueError as e:
        st.error("Invalid server response format")
    except Exception as e:
        st.error("⚠️ An error occurred while processing your question")
        if DEBUG:
            st.exception(e)





# if st.button("Submit") and user_input:
#     response = requests.post(
#         "http://localhost:8000/chat",
#         json={"message": user_input}
#     )
#     data = response.json()
#     st.markdown(f"**Intent:** {data['intent']}")
#     st.markdown(f"**Bot:** {data['response']}")
#     print(data['response'])

#     # Save conversation to database
#     conversations.insert_one({
#         "user": user_input,
#         "bot": data['response']
#     })















# import streamlit as st
# import requests
# import time

# st.set_page_config(page_title="🧠 Project Chatbot", page_icon="🤖")
# st.title("🤖 Chat with Your Project Portfolio")
# st.markdown("Ask about project details, achievements, tabs/pages, typography, etc.")

# # Initialize session state
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Sidebar filters for target-specific queries
# st.sidebar.title("🔍 Optional Target Filters")
# target_option = st.sidebar.selectbox(
#     "What do you want to know specifically?",
#     ["auto (let bot decide)", "achievements", "tabs", "pages", "category", "typography", "color_palette"],
#     index=0
# )

# # User input
# user_input = st.text_input("Type your question:", key="user_input")

# # Send query to backend with optional target
# if st.button("Send") and user_input.strip():
#     st.session_state.history.append({"role": "user", "message": user_input})

#     query_payload = {"query": user_input}
#     if target_option != "auto (let bot decide)":
#         query_payload["target_override"] = target_option

#     with st.spinner("🤖 Generating response..."):
#         try:
#             # Simulated streaming via polling chunks
#             response = requests.post("http://localhost:8001/chat", json=query_payload, stream=True)
#             full_response = ""
#             for chunk in response.iter_content(chunk_size=128):
#                 if chunk:
#                     part = chunk.decode("utf-8")
#                     full_response += part
#                     with st.empty():
#                         st.markdown(f"**🤖 Bot (streaming):** {full_response}")
#             final = full_response.strip()
#         except Exception as e:
#             final = f"❌ Error: {str(e)}"

#         st.session_state.history.append({"role": "bot", "message": final})

# # Display full chat history
# st.divider()
# st.markdown("### 💬 Chat History")
# for msg in st.session_state.history:
#     if msg["role"] == "user":
#         st.markdown(f"**🧑 You:** {msg['message']}")
#     else:
#         st.markdown(f"**🤖 Bot:** {msg['message']}")
