This project enables real-time evaluation of LLM-generated answers using the RAGAS framework. It integrates Gemini (OpenAI-compatible) models with a Qdrant vector store to retrieve context and evaluate responses using metrics like faithfulness, relevancy, and context precision.

# Key Features
Real-time RAG Pipeline: Enter a query, retrieve top-k results from Qdrant, generate an LLM response using Gemini.
RAGAS Evaluation: Computes faithfulness, relevancy, precision, recall, and context relevance.
Visualization: Automatically generates a bar chart of metrics for each evaluated query.
Ground Truth Comparison: Evaluates against human-annotated answers stored in Hugging Face or JSON.

| File                            | Purpose                                                         |
| ------------------------------- | --------------------------------------------------------------- |
| `ragas_live_from_local_json.py` | Loads query-ground truth pairs from `final_dataset.json`        |
| `ragas_live_tester_gemini.py`   | Loads dataset from Hugging Face Hub (via `load_dataset()`)      |
| `qdrant.py`                     | Utility module to manage Qdrant setup and vector search         |
| `live_response_log.json`        | Output log storing all user queries, responses, and evaluations |
| `ragas_live_metrics.png`        | Plot showing evaluation metrics for the last query              |

# Example Interaction
Real-time Gemini RAG Evaluation Mode (type 'exit' to quit)

Enter a user query: What is context in gamification?

 Top Contexts:
  [1] Contextual information from PDF or Qdrant chunk...
  [2] ...

Gemini Response:
Context in gamification refers to...

 Ground Truth Found → Running RAGAS Evaluation...

| answer_relevancy | faithfulness | context_precision | context_recall | context_relevance |
|------------------|--------------|-------------------|----------------|-------------------|
| 0.91             | 0.87         | 0.78              | 0.82           | 0.84              |

 Response saved to live_response_log.json

# Metrics Explained (via RAGAS)
Faithfulness: Is the answer factually grounded in the context?
Answer Relevancy: Does the answer directly address the question?
Context Precision: How much of the context was actually useful?
Context Recall: How much relevant context was retrieved?
Context Relevance: Was the selected context relevant at all?
