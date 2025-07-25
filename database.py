


from fastapi import FastAPI, Request
from pymongo import MongoClient

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

# development
if config.WORKING_ENVIRONMENT == "development":
    mongofbURL = "mongodb+srv://seo-dev:fxo2R9A531Wz80U6@tech-team-dev-mongodb-server-3b7cdaa2.mongo.ondigitalocean.com/seo-dev?tls=true&authSource=admin&replicaSet=tech-team-dev-mongodb-server"

else:
    # Production
    mongofbURL = 'mongodb+srv://jo-dev:E5rjn2IY948t607U@db-mongodb-production-2024-80aa2234.mongo.ondigitalocean.com/jo-dev?tls=true&authSource=admin&replicaSet=db-mongodb-production-2024'



dbClient = MongoClient(mongofbURL, 8000)
db = dbClient["profile_portal"]


