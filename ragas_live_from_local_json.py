import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import sys
sys.path.append(".")
from qdrant import QdrantManager
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    ContextRelevance
)
from openai import OpenAI
from datasets import Dataset

# === Load environment variables ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ragas")

# === Gemini as OpenAI-compatible client ===
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="XYZ" # Replace with actual Gemini API base URL if Needed
    
)

# === Load JSON Dataset ===
with open("final_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

hf_dict = {row['query'].strip().lower(): row['ground_truth'] for row in raw_data}

# === Qdrant Setup ===
qdrant = QdrantManager(collection_name=QDRANT_COLLECTION)

# === Helper Functions ===
def get_context(query, top_k=3):
    results = qdrant.search(query, limit=top_k)
    return [r["text"] for r in results] if results else [""]

def get_gemini_response(query, context):
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering only from context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def evaluate_single(query, ground_truth, response, contexts):
    data = Dataset.from_dict({
    "query": [query],
    "user_input": [query],  # ✅ required by some metrics
    "ground_truth": [ground_truth],
    "response": [response],
    "contexts": [contexts]
})


    result = evaluate(data, metrics=[
        AnswerRelevancy(),
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
        ContextRelevance()
    ])
    return result.to_pandas()

def plot_metrics(df):
    metrics = df.drop("query", axis=1).iloc[0]
    metrics.plot(kind="bar", color="orchid")
    plt.ylim(0, 1.1)
    plt.title("RAGAS Metrics for Query")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("ragas_live_metrics.png")
    plt.show()

# === Main Interactive Loop ===
response_log = []
print("\n🎯 Real-time Gemini RAG Evaluation Mode (type 'exit' to quit)\n")
while True:
    query = input("🔍 Enter a user query: ").strip()
    if query.lower() == "exit":
        break

    contexts = get_context(query)
    context_str = " ".join(contexts)
    response = get_gemini_response(query, context_str)

    print("\n📚 Top Contexts:")
    for i, ctx in enumerate(contexts):
        print(f"  [{i+1}] {ctx[:150]}...")

    print(f"\n🤖 Gemini Response:\n{response}\n")

    gt = hf_dict.get(query.strip().lower())
    if gt:
        print("🎯 Ground Truth Found → Running RAGAS Evaluation...")
        df = evaluate_single(query, gt, response, contexts)
        print(df.to_markdown(index=False))
        plot_metrics(df)

        response_log.append({
            "query": query,
            "ground_truth": gt,
            "llm_response": response
        })
    else:
        print("⚠️ Ground truth not found in dataset — skipping RAGAS evaluation.")
        response_log.append({
            "query": query,
            "ground_truth": None,
            "llm_response": response
        })

    with open("live_response_log.json", "w", encoding="utf-8") as f:
        json.dump(response_log, f, indent=2, ensure_ascii=False)

    print("\n✅ Response saved to live_response_log.json")
    print("="*60 + "\n")
