# RAGAS evaluation script - standalone, run manually via `python evaluate.py`.
# Not part of the live app/agent pipeline. Internal quality-check only.

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from retrieve import load_index, retrieve
from agent import answer_policy_question

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


TEST_SET = [
    {
        "question": "What is the room rent capping under the Basic plan?",
        "tier_for_context_only": "basic",
        "reference": "Room rent is capped at 1% of Sum Insured per day under the Basic plan.",
    },
    {
        "question": "I am a Standard policy holder. Is heart disease covered under my policy?",
        "tier_for_context_only": "standard",
        "reference": "Heart disease treatment is covered under the Standard plan, subject to the applicable waiting periods and sub-limits.",
    },
    {
        "question": "What is the co-pay percentage under the Basic plan?",
        "tier_for_context_only": "basic",
        "reference": "The co-pay under the Basic plan is 10%.",
    },
    {
        "question": "How many times can the Sum Insured be restored under the Premium plan?",
        "tier_for_context_only": "premium",
        "reference": "Under the Premium plan, the Sum Insured can be restored an unlimited number of times per year.",
    },
    {
        "question": "What is the initial waiting period before a claim can be made?",
        "tier_for_context_only": None,
        "reference": "The initial waiting period is 30 days, and it applies to all tiers.",
    },
]


# Building eval dataset
def build_eval_dataset() -> Dataset:
    index = load_index()

    questions, contexts, answers, references = [], [], [], []

    for item in TEST_SET:
        question = item["question"]
        print(f"Evaluating: {question}")

        retrieved_nodes, _ = retrieve(question, index, tier=item["tier_for_context_only"])
        retrieved_texts = [n.get_content() for n in retrieved_nodes]

        # Full pipeline call - classify_query determines tier itself here,
        # exactly as it would for a real user message
        result = answer_policy_question(question)
        generated_answer = result["answer"]

        questions.append(question)
        contexts.append(retrieved_texts)
        answers.append(generated_answer)
        references.append(item["reference"])

    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "reference": references,
    })


# Run RAGAS metrics against the built dataset
def run_evaluation():
    dataset = build_eval_dataset()

    llm = LlamaIndexLLMWrapper(Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=500.0))
    embeddings = LlamaIndexEmbeddingsWrapper(HuggingFaceEmbedding(model_name=EMBEDDING_MODEL))

    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== RAGAS Results ===")
    print(results)
    return results


if __name__ == "__main__":
    run_evaluation()
