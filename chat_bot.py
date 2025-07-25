





from fastapi import FastAPI, Request
# from pymongo import MongoClient
from database import db  # Import the db object from database.py
import subprocess
import json
import config
from datetime import datetime

app = FastAPI()

# # MongoDB Connection
# if config.WORKING_ENVIRONMENT == "development":
#     mongofbURL = "mongodb+srv://seo-dev:fxo2R9A531Wz80U6@tech-team-dev-mongodb-server-3b7cdaa2.mongo.ondigitaloceanspaces.com/seo-dev?tls=true&authSource=admin&replicaSet=tech-team-dev-mongodb-server"
#     # mongofbURL = "mongodb://seo-dev:fxo2R9A531Wz80U6@tech-team-dev-mongodb-server-3b7cdaa2.mongo.ondigitaloceanspaces.com:27017/seo-dev?tls=true&authSource=admin&replicaSet=tech-team-dev-mongodb-server"

# else:
#     mongofbURL = "mongodb+srv://jo-dev:E5rjn2IY948t607U@db-mongodb-production-2024-80aa2234.mongo.ondigitaloceanspaces.com/jo-dev?tls=true&authSource=admin&replicaSet=db-mongodb-production-2024"
#     # mongofbURL = "mongodb://seo-dev:fxo2R9A531Wz80U6@tech-team-dev-mongodb-server-3b7cdaa2.mongo.ondigitaloceanspaces.com:27017/seo-dev?tls=true&authSource=admin&replicaSet=tech-team-dev-mongodb-server"

# # development
# if config.WORKING_ENVIRONMENT == "development":
#     mongofbURL = "mongodb+srv://seo-dev:fxo2R9A531Wz80U6@tech-team-dev-mongodb-server-3b7cdaa2.mongo.ondigitalocean.com/seo-dev?tls=true&authSource=admin&replicaSet=tech-team-dev-mongodb-server"

# else:
#     # Production
#     mongofbURL = 'mongodb+srv://jo-dev:E5rjn2IY948t607U@db-mongodb-production-2024-80aa2234.mongo.ondigitalocean.com/jo-dev?tls=true&authSource=admin&replicaSet=db-mongodb-production-2024'



# dbClient = MongoClient(mongofbURL, 8000)
# db = dbClient["profile_portal"]

OLLAMA_MODEL = "mistral:latest"

# 🧠 Helper to run Ollama
def ollama_chat(prompt: str):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

# 🧠 Extract query info using Ollama
def extract_query_info(user_query: str):
    prompt = f"""
        You are a project assistant. Extract structured intent from this query:
        "{user_query}"

        Respond in this JSON format:
        {{
        "intent": "by_name | by_client | by_category | by_type | by_scope | unknown",
        "project_name": null,
        "client_name": null,
        "category": null,
        "website_type": null,
        "scope": null,
        "target": "achievements | tabs | pages | category | typography | color_palette | all | unknown"
        }}
        Only include values mentioned clearly. Do not guess.
        """
    raw = ollama_chat(prompt)
    try:
        json_start = raw.find('{')
        return json.loads(raw[json_start:])
    except Exception:
        return {"intent": "unknown", "target": "unknown"}

# 🧱 Datetime converter for JSON safety
def convert_datetime(obj):
    if isinstance(obj, list):
        return [convert_datetime(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

# 🧩 Enrich a single project
def enrich_project(project):
    project_id = int(project["id"])

    project["category"] = db.categories.find_one({"id": project.get("category_id")}, {"_id": 0}) or {}
    project["website_type"] = db.website_types.find_one({"id": project.get("website_type_id")}, {"_id": 0}) or {}

    scope_ids = project.get("scope_of_work_ids", [])
    project["scope_of_work"] = list(db.scope_of_work.find({"id": {"$in": scope_ids}}, {"_id": 0}))

    project["achievements"] = list(db.achievements.find({"project_id": project_id, "deleted": {"$ne": 1}}, {"_id": 0}))
    project["testimonials"] = list(db.testimonials.find({"project_id": str(project_id), "deleted": {"$ne": 1}}, {"_id": 0}))
    project["comments"] = list(db.comments.find({"project_id": project_id, "deleted": {"$ne": 1}}, {"_id": 0}))

    tabs = list(db.tabs.find({"project_id": str(project_id), "deleted": {"$ne": 1}}, {"_id": 0}))
    for tab in tabs:
        tab["pages"] = list(db.pages.find({"tab_id": str(tab["id"]), "deleted": {"$ne": 1}}, {"_id": 0}))
    project["tabs"] = tabs

    for field in ["typography", "color_palette"]:
        try:
            if isinstance(project.get(field), str):
                project[field] = json.loads(project[field])
        except:
            pass

    return project

# 📡 Main Chat Endpoint
@app.post("/chat")
async def chatbot(request: Request):
    body = await request.json()
    user_query = body.get("query", "")
    parsed = extract_query_info(user_query)

    intent = parsed.get("intent")
    target = parsed.get("target", "all") or "all"
    filtered_projects = []

    # print(f"Parsed intent: {intent}, data: {parsed}")

    base_query = {"deleted": {"$ne": 1}}

    if intent == "by_name" and parsed.get("project_name"):
        base_query["name"] = {"$regex": f"^{parsed['project_name']}$", "$options": "i"}
        projects = list(db.projects.find(base_query, {"_id": 0}))
        filtered_projects = [enrich_project(p) for p in projects]

    elif intent == "by_client" and parsed.get("client_name"):
        base_query["client_name"] = {"$regex": f"^{parsed['client_name']}$", "$options": "i"}
        projects = list(db.projects.find(base_query, {"_id": 0}))
        filtered_projects = [enrich_project(p) for p in projects]

    elif intent in ["by_category", "by_type", "by_scope"]:
        projects = list(db.projects.find(base_query, {"_id": 0}))
        for p in projects:
            enriched = enrich_project(p)
            if (
                intent == "by_category" and enriched.get("category", {}).get("name", "").lower() == parsed["category"].lower()
            ) or (
                intent == "by_type" and enriched.get("website_type", {}).get("name", "").lower() == parsed["website_type"].lower()
            ) or (
                intent == "by_scope" and any(
                    s["name"].lower() == parsed["scope"].lower()
                    for s in enriched.get("scope_of_work", [])
                )
            ):
                filtered_projects.append(enriched)

    if not filtered_projects:
        return {"response": "❌ No matching projects found."}

    filtered_projects = convert_datetime(filtered_projects)

    short_prompt = ""
    if target == "achievements":
        short_prompt = "List the achievements (title, description, cards) for the matched project."
    elif target == "tabs":
        short_prompt = "List the tabs with titles and descriptions."
    elif target == "pages":
        short_prompt = "List the pages grouped by tab with titles and descriptions."
    elif target == "category":
        short_prompt = "State the category of the project."
    elif target == "typography":
        short_prompt = "Describe the typography fonts used."
    elif target == "color_palette":
        short_prompt = "Describe the primary and secondary color palettes."
    else:
        short_prompt = (
            "Summarize the full project: name, client, HQ, timeline, type, category, scope, "
            "typography, color palette, tabs/pages, testimonials, achievements."
        )


    answer_prompt = f"""
        You are an intelligent and friendly AI assistant that helps users understand project portfolio data.

        The user asked:
        \"\"\"{user_query}\"\"\"

        Below is the structured project data you should use to answer:
        {json.dumps(convert_datetime(filtered_projects), indent=2)}

        Your job:
        - Only use the information provided in the project data.
        - Never make up or hallucinate missing values.
        - Follow the specific instruction below to determine what kind of response to return.


        Your task:
        - Use the project data to answer the user's question.
        - Follow this instruction: {short_prompt}

        Rules:
        - Only use the provided project data. Do not hallucinate or assume.
        - Format the answer as a natural, well-written paragraph.
        - Use complete sentences and friendly language.
        - Do not output JSON, raw values, or bullet points.
        - Imagine you're explaining this to a non-technical client in a professional tone.

        Begin your answer now:

        Now generate the best possible answer:
    """





    final_response = ollama_chat(answer_prompt)
    return {"response": final_response.strip()}


