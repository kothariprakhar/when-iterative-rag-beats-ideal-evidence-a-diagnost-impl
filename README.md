# When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering

Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks.

## Implementation Details

# Implementation Explanation: Iterative RAG Diagnostic Study

## 1. Brainstorming & Design Choices
To replicate the findings of the paper "When Iterative RAG Beats Ideal Evidence," we need a controlled environment that isolates the effects of **retrieval dynamics** versus **context availability**.

*   **Architecture Choice**: We utilize a **Retrieve-Then-Generate** architecture. For the LLM, `google/flan-t5-base` is selected. This model is small enough for local experimentation but trained on instructions, making it capable of following the "generate a query" vs "answer the question" distinction. For retrieval, we use `sentence-transformers` (specifically `all-MiniLM-L6-v2`) combined with `FAISS`. This provides a dense vector index capable of semantic search, which is crucial for multi-hop questions where keyword overlap is minimal.
*   **Simulating the Scientific Domain**: The paper uses `ChemKGMultiHopQA`. Since this is specialized and potentially proprietary, we use **HotpotQA (Distractor Setting)** as a robust proxy. HotpotQA requires multi-hop reasoning (finding fact A to find fact B) and provides "Gold" supporting facts, allowing us to construct the "Gold Context" baseline perfectly.
*   **Iterative Mechanism**: Instead of a static Retrieve-Read, we implement a loop. The model generates a *search query* based on what it knows, retrieves, updates context, and repeats. This mimics the "hypothesis refinement" described in the paper.
*   **Trade-offs**: A base-sized model (250M params) has limited reasoning capabilities compared to GPT-4. Consequently, the "Iterative" performance might not strictly beat "Gold" in this small-scale demo, but the *code architecture* accurately reflects the paper's mechanism.

## 2. Dataset & Tools
*   **Dataset**: [HotpotQA (HuggingFace)](https://huggingface.co/datasets/hotpot_qa). We use the 'distractor' configuration which includes the question, answer, supporting facts, and a mix of relevant/irrelevant context paragraphs.
*   **Tools**: `transformers` (LLM), `sentence-transformers` (Embeddings), `faiss` (Vector Search), `pandas/seaborn` (Analysis).

## 3. Architecture & Math
The implementation follows the three regimes defined in the paper:

1.  **Parametric (No Context)**: $P(y|x)$. The model relies solely on pre-training weights.
2.  **Gold Context (Ideal)**: $P(y|x, C_{gold})$. The model is given the ground-truth supporting sentences $C_{gold}$. This represents the theoretical upper bound of static RAG.
3.  **Iterative RAG**: Modeled as a state sequence.
    *   At step $t$, context $C_t = C_{t-1} \cup d_t$.
    *   Query Generation: $q_t = \text{LLM}(x, C_{t-1})$.
    *   Retrieval: $d_t = \text{Retriever}(q_t)$.
    *   Update: $C_t$.
    *   This mitigates **Context Overload** (receiving too much noise at once) by focusing attention on one hop at a time.

## 4. Walkthrough
1.  **Preprocessing**: We load HotpotQA. For each question, we construct a "local universe" of documents. In a real system, this would be a global index, but for the 'distractor' setting, the search space is the provided paragraphs (gold + distractors).
2.  **Vector Indexing**: We encode all sentences in this local universe using `SentenceTransformer` and build a FAISS index.
3.  **RAG Loop**: 
    *   The `RAGController` first tries to answer from memory (Regime 1).
    *   Then it answers using provided Gold Facts (Regime 2).
    *   Finally, it enters the Iterative Loop (Regime 3). It prompts the LLM to identify *missing information*, searches the index, and appends the result.
4.  **Evaluation**: We use Exact Match (EM) normalization to compare model output with the ground truth.

## 5. Visuals & Plots
The notebook generates two key plots:
1.  **Accuracy Bar Chart**: Compares the Exact Match (EM) scores of No Context, Gold Context, and Iterative RAG. This visualizes the central claim of the paper—whether Iterative can approach or exceed Gold Context.
2.  **Recall Histogram**: A diagnostic plot showing the distribution of Gold Fact Recall for the Iterative agent. High recall indicates the iterative queries successfully navigated the multi-hop chain; low recall explains failure cases (e.g., getting stuck on a distractor).

## Verification & Testing

The code implements the core logic of Iterative RAG and correctly handles the HotpotQA dataset structure for the 'distractor' setting (constructing a local retrieval pool per question). 

**Strengths:**
- **Logic:** The iterative loop (Generation $\rightarrow$ Retrieval $\rightarrow$ Update Context $\rightarrow$ Stop Check) faithfully reproduces the generic Iterative RAG workflow.
- **Dataset Handling:** Correctly parses HotpotQA's nested `context` and `supporting_facts` to derive gold evidence and retrieval corpus.
- **Metrics:** Implements standard Exact Match (EM) normalization and calculation.

**Weaknesses & Potential Issues:**
- **Efficiency:** Rebuilding the FAISS index for every single question (`build_index` inside the loop) is computationally expensive, though necessary here to simulate the 'distractor' setting where the search space is local to the question's candidate paragraphs.
- **Type Safety:** FAISS usually expects `float32` input. While `SentenceTransformer` often returns `float32`, explicit casting (`astype('float32')`) before passing to `index.add` is best practice to avoid type errors on some platforms.
- **Context Window:** `Flan-T5-base` has a functional context window (typically 512 tokens). Appending multiple documents iteratively might exceed this limit. The code sets `max_length=1024` which T5 might handle via relative positions, but performance degrades or truncation occurs silently.
- **Hardcoded Prompts:** The prompts are hardcoded. While functional, they are brittle to model changes. 
- **Stopping Heuristic:** The boolean check (`"yes" in can_answer.lower()`) is a simple heuristic that works for `Flan-T5` but may yield false positives if the model outputs "Yesterday..." for a temporal question.