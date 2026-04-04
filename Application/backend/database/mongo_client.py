"""
CaneNexus – MongoDB Connection Singleton
Provides a shared database handle and collection references.
"""

from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB

# ── Connection ──
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]

# ── Collections ──
logs_collection = db["logs"]                # Individual image analysis logs
simulations_collection = db["simulations"]  # Simulation session metadata
chats_collection = db["chats"]              # Chat conversation histories
feedback_collection = db["feedback_corrections"]  # Operator feedback / corrections
finetune_jobs_collection = db["finetune_jobs"]    # Fine-tune job metadata
model_versions_collection = db["model_versions"]  # Model checkpoint version registry


def check_connection():
    """
    Verify that MongoDB is reachable.
    Returns True if connected, False otherwise.
    """
    try:
        client.admin.command("ping")
        return True
    except Exception:
        return False
