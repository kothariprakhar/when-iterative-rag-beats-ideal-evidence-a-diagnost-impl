import torch
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. Configuration & Setup
# ==========================================
MODEL_NAME = "google/flan-t5-base"  # Efficient instruction-tuned model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATASET_NAME = "hotpot_qa"
DATASET_CONFIG = "distractor"
SAMPLE_SIZE = 30  # Number of samples for the demo to run quickly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on device: {DEVICE}")

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print("Loading dataset...")
# We use HotpotQA as a proxy for multi-hop scientific QA due to its structure.
# The 'distractor' config provides hard negatives, simulating heterogeneous evidence.
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation", trust_remote_code=True)
dataset = dataset.select(range(SAMPLE_SIZE))

# ==========================================
# 3. Retrieval System (FAISS)
# ==========================================
class DenseRetriever:
    def __init__(self, embedding_model_name):
        self.encoder = SentenceTransformer(embedding_model_name, device=DEVICE)
        self.index = None
        self.docs = []
    
    def build_index(self, corpus_texts):
        """Builds a FAISS index for the provided corpus."""
        self.docs = corpus_texts
        print(f"Encoding {len(corpus_texts)} documents...")
        embeddings = self.encoder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize for IP (Cosine)
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
    def retrieve(self, query, k=1):
        """Retrieves top-k documents for a query."""
        q_embed = self.encoder.encode([query], convert_to_numpy=True)
        q_embed = q_embed / np.linalg.norm(q_embed, axis=1, keepdims=True)
        D, I = self.index.search(q_embed, k)
        return [self.docs[i] for i in I[0]], D[0]

# ==========================================
# 4. Agent Architecture (Iterative RAG)
# ==========================================
class RAGController:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        
    def generate(self, prompt, max_new_tokens=64):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_no_context(self, question):
        """Regime 1: No Context (Parametric Memory)"""
        prompt = f"Question: {question}\nAnswer:"
        return self.generate(prompt)

    def run_gold_context(self, question, gold_facts):
        """Regime 2: Gold Context (Ideal Evidence)"""
        # Join all gold facts at once
        context_str = " ".join(gold_facts)
        prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
        return self.generate(prompt)

    def run_iterative_rag(self, question, retriever, max_steps=3):
        """Regime 3: Iterative RAG with Hypothesis Refinement"""
        current_context = []
        trajectory = []
        
        for step in range(max_steps):
            # 1. Hypothesis/Query Generation Step
            context_str = " ".join(current_context)
            # Prompt engineering to encourage reasoning about what is missing
            if step == 0:
                reasoning_prompt = (
                    f"Question: {question}\n"
                    "Task: Identify the first entity or fact needed to answer this question. Output a search query."
                )
            else:
                reasoning_prompt = (
                    f"Question: {question}\n"
                    f"Known Facts: {context_str}\n"
                    "Task: Based on Known Facts, what information is still missing? Output a search query for the missing info."
                )
            
            search_query = self.generate(reasoning_prompt, max_new_tokens=32)
            
            # 2. Retrieval Step
            # We search in the specific pool of paragraphs available for this question (local pool for HotpotQA)
            # In a real open-domain scenario, this searches the whole wiki.
            docs, scores = retriever.retrieve(search_query, k=1)
            retrieved_doc = docs[0]
            
            # Avoid duplicates
            if retrieved_doc not in current_context:
                current_context.append(retrieved_doc)
                
            trajectory.append({"step": step, "query": search_query, "retrieved": retrieved_doc})
            
            # 3. Check for Stopping/Answer Readiness
            # A simple heuristic: ask model if it can answer.
            check_prompt = (
                f"Question: {question}\n"
                f"Context: {' '.join(current_context)}\n"
                "Can you answer the question based on the context? Answer yes or no."
            )
            can_answer = self.generate(check_prompt, max_new_tokens=5)
            
            if "yes" in can_answer.lower():
                break
        
        # Final Generation
        final_prompt = f"Context: {' '.join(current_context)}\nQuestion: {question}\nAnswer:"
        final_answer = self.generate(final_prompt)
        return final_answer, trajectory, current_context

# ==========================================
# 5. Experiment Execution
# ==========================================
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re, string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def run_benchmark():
    # Initialize modules
    retriever_model = DenseRetriever(EMBEDDING_MODEL)
    agent = RAGController(MODEL_NAME)
    
    results = []
    
    print("Starting Benchmark...")
    for idx, item in tqdm(enumerate(dataset), total=SAMPLE_SIZE):
        question = item['question']
        answer = item['answer']
        
        # Prepare local corpus for this question (HotpotQA structure)
        # Item['context'] is a list of [title, sentences]
        # We flatten this to create a retrieval pool for the 'distractor' setting
        corpus_texts = []
        gold_facts = []
        
        # Map titles to support facts
        # supporting_facts is list of [title, sent_id]
        sup_facts_set = set([(f[0], f[1]) for f in item['supporting_facts']])
        
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            for i, sent in enumerate(sentences):
                full_sent = f"{title}: {sent}"
                corpus_texts.append(full_sent)
                if (title, i) in sup_facts_set:
                    gold_facts.append(full_sent)
        
        # Build Index for this specific question's universe (simulating a large DB subset)
        retriever_model.build_index(corpus_texts)
        
        # 1. No Context
        ans_no_ctx = agent.run_no_context(question)
        em_no_ctx = exact_match_score(ans_no_ctx, answer)
        
        # 2. Gold Context (Oracle)
        ans_gold = agent.run_gold_context(question, gold_facts)
        em_gold = exact_match_score(ans_gold, answer)
        
        # 3. Iterative RAG
        ans_iter, traj, final_ctx = agent.run_iterative_rag(question, retriever_model)
        em_iter = exact_match_score(ans_iter, answer)
        
        # Calculate Retrieval Recall for Iterative
        # What % of gold facts ended up in the final context?
        gold_set = set(gold_facts)
        retrieved_set = set(final_ctx)
        recall = len(gold_set.intersection(retrieved_set)) / len(gold_set) if len(gold_set) > 0 else 0
        
        results.append({
            "question_id": idx,
            "no_context_em": em_no_ctx,
            "gold_em": em_gold,
            "iterative_em": em_iter,
            "iterative_recall": recall,
            "iter_steps": len(traj)
        })
        
    return pd.DataFrame(results)

# ==========================================
# 6. Visualization
# ==========================================
if __name__ == "__main__":
    df_results = run_benchmark()
    
    # Metrics Summary
    print("\nResults Summary:")
    print(df_results.mean(numeric_only=True))
    
    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(10, 6))
    means = df_results[['no_context_em', 'gold_em', 'iterative_em']].mean()
    sns.barplot(x=means.index, y=means.values, palette="viridis")
    plt.title("Exact Match Accuracy across Regimes")
    plt.ylabel("Accuracy (EM)")
    plt.xlabel("Regime")
    plt.ylim(0, 1.0)
    plt.show()
    
    # Plot 2: Iterative Recall Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_results['iterative_recall'], bins=5, kde=True, color='orange')
    plt.title("Distribution of Gold Evidence Recall in Iterative RAG")
    plt.xlabel("Recall of Gold Facts")
    plt.show()
