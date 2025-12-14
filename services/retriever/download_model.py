import os
import shutil
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
OUTPUT_DIR = "./model_data"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

print(f"📉 Downloading {MODEL_NAME} (Rev: {REVISION})...")

try:
    model = SentenceTransformer(MODEL_NAME, revision=REVISION)
    model.save(OUTPUT_DIR)
    print(f"✅ Model baked into {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    exit(1)