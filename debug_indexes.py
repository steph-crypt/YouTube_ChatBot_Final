# debug_indexes.py
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Your available indexes:")
print(pc.list_indexes().names())