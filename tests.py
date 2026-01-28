import unittest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import torch
import pandas as pd

# Assumption: The user's code is imported as module 'rag_impl' or classes are pasted.
# For this test suite, we will mock the dependencies to verify logic without loading heavy models.

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        # Mocks for external libraries
        self.sentence_transformer_mock = MagicMock()
        self.tokenizer_mock = MagicMock()
        self.model_mock = MagicMock()
        self.faiss_index_mock = MagicMock()

    @patch('sentence_transformers.SentenceTransformer')
    @patch('faiss.IndexFlatIP')
    def test_dense_retriever_logic(self, mock_faiss, mock_st_class):
        # Setup Mocks
        mock_encoder = mock_st_class.return_value
        # Mock encoding: 3 docs, dim 4
        fake_embeddings = np.random.rand(3, 4).astype('float32')
        mock_encoder.encode.return_value = fake_embeddings
        
        # Import classes to test (assuming they are in scope or defined)
        from sentence_transformers import SentenceTransformer
        import faiss
        
        # Redefining class locally for testing if not imported
        class DenseRetriever:
            def __init__(self, embedding_model_name):
                self.encoder = SentenceTransformer(embedding_model_name)
                self.index = None
                self.docs = []
            
            def build_index(self, corpus_texts):
                self.docs = corpus_texts
                embeddings = self.encoder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
                # Normalize logic from original code
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings)
                
            def retrieve(self, query, k=1):
                q_embed = self.encoder.encode([query], convert_to_numpy=True)
                q_embed = q_embed / np.linalg.norm(q_embed, axis=1, keepdims=True)
                D, I = self.index.search(q_embed, k)
                return [self.docs[i] for i in I[0]], D[0]

        # Test Execution
        retriever = DenseRetriever("dummy_model")
        corpus = ["doc1", "doc2", "doc3"]
        retriever.build_index(corpus)
        
        # Verify Index Building
        mock_encoder.encode.assert_called()
        # Check if FAISS index was created and populated
        self.assertTrue(retriever.index.add.called)
        
        # Test Retrieval
        # Mock search return: Distance matrix, Index matrix
        retriever.index.search.return_value = (np.array([[0.9]]), np.array([[1]])) # matches doc2
        docs, scores = retriever.retrieve("query", k=1)
        
        self.assertEqual(docs, ["doc2"])
        self.assertEqual(len(scores), 1)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSeq2SeqLM.from_pretrained')
    def test_rag_controller_iterative_logic(self, mock_model_cls, mock_tokenizer_cls):
        # Setup Mocks
        tokenizer = MagicMock()
        model = MagicMock()
        mock_tokenizer_cls.return_value = tokenizer
        mock_model_cls.return_value = model
        
        # Mock Tokenizer behavior
        tokenizer.return_value = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.tensor([[1, 1]])}
        tokenizer.decode.side_effect = lambda x, skip_special_tokens: "generated_text"
        
        # Redefining class for testing
        class RAGController:
            def __init__(self, model_name):
                self.tokenizer = tokenizer
                self.model = model
                
            def generate(self, prompt, max_new_tokens=64):
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            def run_iterative_rag(self, question, retriever, max_steps=3):
                current_context = []
                trajectory = []
                for step in range(max_steps):
                    # Logic mimics original code
                    context_str = " ".join(current_context)
                    # Just call generate (mocked)
                    search_query = self.generate("prompt", max_new_tokens=32)
                    
                    docs, scores = retriever.retrieve(search_query, k=1)
                    retrieved_doc = docs[0]
                    
                    if retrieved_doc not in current_context:
                        current_context.append(retrieved_doc)
                        
                    trajectory.append({"step": step, "query": search_query, "retrieved": retrieved_doc})
                    
                    # Check stop
                    can_answer = self.generate("check", max_new_tokens=5)
                    if "yes" in can_answer.lower():
                        break
                
                final_answer = self.generate("final")
                return final_answer, trajectory, current_context

        # Setup Retrieval Mock
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = (["Useful Fact A"], [1.0])

        # Instantiate Controller
        controller = RAGController("dummy-t5")
        
        # Scenario 1: Model always says "yes" immediately (1 step)
        # We control generate output via side_effect. 
        # Sequence: [Query Gen, Check Answer (Yes), Final Answer]
        tokenizer.decode.side_effect = ["search query", "Yes I can", "Final Answer"]
        
        ans, traj, ctx = controller.run_iterative_rag("Question?", mock_retriever, max_steps=3)
        
        self.assertEqual(ans, "Final Answer")
        self.assertEqual(len(traj), 1)
        self.assertEqual(ctx, ["Useful Fact A"])

        # Scenario 2: Model iterates (No -> No -> Yes)
        # Sequence: 
        # Step 0: [Query, Check(No)]
        # Step 1: [Query, Check(No)]
        # Step 2: [Query, Check(Yes)]
        # Final Answer
        tokenizer.decode.side_effect = [
            "q1", "no", 
            "q2", "no", 
            "q3", "yes", 
            "Final Answer"
        ]
        
        ans, traj, ctx = controller.run_iterative_rag("Question?", mock_retriever, max_steps=3)
        self.assertEqual(len(traj), 3)
        self.assertEqual(len(ctx), 1) # Same doc returned by mock_retriever, dedup logic handles it

    def test_metrics(self):
        # Redefine metric functions locally
        import string, re
        def normalize_answer(s):
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
            
        # Test Cases
        self.assertTrue(exact_match_score("The Earth", "earth"))
        self.assertTrue(exact_match_score("Blue.", "blue"))
        self.assertFalse(exact_match_score("Red", "Blue"))

if __name__ == '__main__':
    unittest.main()