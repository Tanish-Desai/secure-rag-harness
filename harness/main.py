import os
import fire
import json
import time
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Configuration
GATEWAY_URL = "http://localhost:8000/chat"

def run_experiment(
    attack="none",
    defense="baseline",
    profile="P1",
    topology="sequential",
    seed=42,
    limit=10,
    output_dir="results"
):
    """
    Orchestrates a Secure RAG Experiment.
    """
    print(f"Starting Experiment: Attack={attack} | Defense={defense} | Profile={profile}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 1. Load Test Data (For now, we generate synthetic queries on the fly)
    # In the next step, we will load real datasets
    queries = [
        f"Query {i}: What is fact {i}?" for i in range(limit)
    ]
    
    results = []
    
    # 2. Main Loop
    for q in tqdm(queries, desc="Running Queries"):
        start_time = time.time()
        
        payload = {
            "query": q,
            "topology": topology,
            "profile": profile,
            "seed": seed
        }
        
        try:
            # Send to Gateway
            resp = requests.post(GATEWAY_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            latency = time.time() - start_time
            
            results.append({
                "query": q,
                "response": data["response"],
                "context_used": data["context_used"],
                "latency_seconds": latency,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "query": q,
                "error": str(e),
                "status": "error"
            })

    # 3. Save Results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/exp_{attack}_{defense}_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"\nExperiment Complete. Results saved to {filename}")
    print(df.head())

if __name__ == "__main__":
    fire.Fire(run_experiment)