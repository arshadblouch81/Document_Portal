# tests/test_unit_cases.py


import pytest
from fastapi.testclient import TestClient
# from api.main import app   # or your FastAPI entrypoint
from api.main import app
import BytesIO

from langchain.evaluation.qa import QAEvalChain

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
# from deepeval.dataset import EvaluationDataset, Golden
# from deepeval.integrations.langchain import CallbackHandler

from datasets import Dataset 
# from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness


client = TestClient(app)
# callback_handler = CallbackHandler()

#--------------Test Case-1-----------------------------
# Test case to check login page
def test_login():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text
#-----------------Test Case-2-----------------------------
# Test case to check document analysis
# def test_analyze_document_success():
#     # Simulate a small text file upload
#     test_file_content = b"This is a sample document for DeepEval and analysis testing."
#     files = {"file": ("test.txt", BytesIO(test_file_content), "text/plain")}

#     response = client.post("/analyze", files=files)

#     # Basic assertions
#     assert response.status_code == 200
#     data = response.json()
#     assert isinstance(data, dict)

#     # Validate merged keys from result and deepeval
#     # Adjust these keys based on actual structure returned by analyze_document and find_deepEval
#     expected_result_keys = ["summary", "Title", "Author"]  # Example keys from result
#     expected_deepeval_keys = ["score", "metrics", "explanation"]  # Example keys from deepeval

#     for key in expected_result_keys + expected_deepeval_keys:
#         assert key in data
# #-----------------Test Case-3-----------------------------
# #Test case to check document comparison
# def test_compare_documents_success():
#     # Simulate two text files for comparison
#     reference_content = b"This is the reference document. It contains baseline information."
#     actual_content = b"This is the actual document. It contains updated information."

#     files = {
#         "reference": ("reference.txt", BytesIO(reference_content), "text/plain"),
#         "actual": ("actual.txt", BytesIO(actual_content), "text/plain")
#     }

#     response = client.post("/compare", files=files)

#     # Basic assertions
#     assert response.status_code == 200
#     data = response.json()
#     assert "rows" in data
#     assert "session_id" in data
#     assert isinstance(data["rows"], list)
#     assert isinstance(data["session_id"], str)

#     # Optional: Validate structure of comparison rows
#     if data["rows"]:
#         row = data["rows"][0]
#         expected_keys = ["reference_text", "actual_text", "difference", "score"]  # Adjust based on actual schema
#         for key in expected_keys:
#             assert key in row

# #----------------Test Case-4-----------------------------
# #Test case to check chat index building

# def test_chat_build_index_success():
#     # Simulate two small text files
#     file1 = ("doc1.txt", BytesIO(b"First document content."), "text/plain")
#     file2 = ("doc2.txt", BytesIO(b"Second document content."), "text/plain")

#     files = [
#         ("files", file1),
#         ("files", file2)
#     ]

#     form_data = {
#         "session_id": "test-session-123",
#         "use_session_dirs": "true",
#         "chunk_size": "512",
#         "chunk_overlap": "128",
#         "k": "3"
#     }

#     response = client.post("/chat/index", files=files, data=form_data)

#     # Basic assertions
#     assert response.status_code == 200
#     data = response.json()
#     assert "session_id" in data
#     assert "k" in data
#     assert "use_session_dirs" in data

#     # Validate values
#     assert data["session_id"] == "test-session-123"
#     assert data["k"] == 3
#     assert data["use_session_dirs"] is True

# #------------------Test case 5-----------------------------
# # Test case to check successful chat query
# def test_chat_query_success(tmp_path, monkeypatch):
#     # Simulate a valid FAISS index directory
#     session_id = "test-session"
#     index_dir = tmp_path / session_id
#     index_dir.mkdir()

#     # Monkeypatch FAISS_BASE to point to temp directory
#     monkeypatch.setattr("main.FAISS_BASE", str(tmp_path))

#     # Mock ConversationalRAG and its methods
#     from unittest.mock import patch, MagicMock

#     with patch("main.ConversationalRAG") as mock_rag_class:
#         mock_rag_instance = MagicMock()
#         mock_rag_instance.invoke.return_value = "This is a mock answer."
#         mock_rag_class.return_value = mock_rag_instance

#         form_data = {
#             "question": "What is semantic search?",
#             "session_id": session_id,
#             "use_session_dirs": "true",
#             "k": "3"
#         }

#         response = client.post("/chat/query", data=form_data)

#         # Assertions
#         assert response.status_code == 200
#         data = response.json()
#         assert data["answer"] == "This is a mock answer."
#         assert data["session_id"] == session_id
#         assert data["k"] == 3
#         assert data["engine"] == "LCEL-RAG"
# #--------------------Test Case-6-----------------------------
# # Test Case to check query with no session ID
# def test_chat_query_missing_session_id():   
#     form_data = {
#         "question": "What is vector search?",
#         "use_session_dirs": "true",
#         "k": "5"
#     }

#     response = client.post("/chat/query", data=form_data)
#     assert response.status_code == 400
#     assert "session_id is required" in response.json()["detail"]

# #--------------------Test Case-7-----------------------------
# # Test Case to check FAISS index not found
# def test_chat_query_index_not_found(monkeypatch):
#     monkeypatch.setattr("main.FAISS_BASE", "/nonexistent/path")

#     form_data = {
#         "question": "Explain embeddings.",
#         "session_id": "ghost-session",
#         "use_session_dirs": "true",
#         "k": "5"
#     }

#     response = client.post("/chat/query", data=form_data)
#     assert response.status_code == 404
#     assert "FAISS index not found" in response.json()["detail"]

# #--------------------Test Case-8-----------------------------

# def deepEval_case_1():
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
#     test_case = LLMTestCase(
#         input="What if these shoes don't fit?",
#         # Replace this with the actual output from your LLM application
#         actual_output="We offer a 30-day full refund at no extra costs.",
#     retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
#     )
#     evaluate([test_case], [answer_relevancy_metric])
# # --------------------Test Case-9-----------------------------

# # def deepEval(self, model, prompts):
# #     # Setup DeepEval
# #     # Choose metrics relevant to your use case
# #     answer_relevancy = AnswerRelevancyMetric(model=model)
# #     contextual_accuracy = ContextualAccuracyMetric(model=model)
# #     test_cases = [
# #     LLMTestCase(
# #         input="What is the capital of France?",
# #         actual_output="The capital of France is Paris.",
# #         expected_output="Paris"
# #     ),
# #     LLMTestCase(
# #         input="Explain Newton's second law.",
# #         actual_output="Newton's second law states that force equals mass times acceleration.",
# #         expected_output="F = ma"
# #     )
# #     ]
# #     for test_case in test_cases:
# #         print("Evaluating:", test_case.input)
# #         relevancy_score = answer_relevancy.measure(test_case)
# #         accuracy_score = contextual_accuracy.measure(test_case)
        
# #         print(f"Answer Relevancy: {relevancy_score.score}")
# #         print(f"Contextual Accuracy: {accuracy_score.score}")
# #         print("-" * 40)
# #     return {
# #         "answer_relevancy": answer_relevancy,
# #         "contextual_accuracy": contextual_accuracy
# #     }

# #--------------------Test Case-9----------------------------- 
# def find_deepEval(self, prompt: str, answer: str):
#     """
#     Find and return relevant DeepEval metrics for the given document text.
#     """
#     try:
#         evaluator = QAEvalChain.from_llm(llm=self.llm)

#         # Properly format examples and predictions
#         examples = [{"query": prompt, "answer": answer}]
#         predictions = [{"answer": answer}]

#         results = evaluator.evaluate(examples=examples, predictions=predictions)

#         self.log.info("DeepEval metrics extraction successful")
#         return results

#     except Exception as e:
#         self.log.error(f"DeepEval metrics extraction failed: {e}")
#         raise Exception("DeepEval metrics extraction failed") from e
    
    
# # -------------------------TestCase 10 --------------------------------------------
# def find_deepEval2(self, context:str, prompt:str, answer:str)-> str:
#     """
#     Find and return relevant DeepEval metrics for the given document text.
#     """
#     try:
#         data_samples = {
#             'question': prompt,
#             'answer': answer,
#             'contexts': context,
#             'ground_truth': [
#                 'The first superbowl was held on January 15, 1967', 
#                 'The New England Patriots have won the Super Bowl a record six times'
#             ]
#         }

#         dataset = Dataset.from_dict(data_samples)

#         score = evaluate(dataset, metrics=[faithfulness, answer_correctness])                       
        
#         self.log.info("DeepEval metrics extraction successful")
#         return score

#     except Exception as e:
#         self.log.error("DeepEval metrics extraction failed", error=str(e))
#         raise Exception("DeepEval metrics extraction failed") from e

