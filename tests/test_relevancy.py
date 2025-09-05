
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from utils.model_loader import ModelLoader
from deepeval.models import DeepEvalBaseLLM

class GoogleGenAIDeepEvalWrapper(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content  # Adjust based on LangChain's response structure


def test_relevancy():
   
    llm = ModelLoader().load_llm()  # returns ChatGoogleGenerativeAI
    wrapped_llm = GoogleGenAIDeepEvalWrapper(llm)

    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=wrapped_llm)
    
    test_case_1 = LLMTestCase(
        input="Can I return these shoes after 30 days?",
        actual_output="Unfortunately, returns are only accepted within 30 days of purchase.",
        retrieval_context=[
            "All customers are eligible for a 30-day full refund at no extra cost.",
            "Returns are only accepted within 30 days of purchase.",
        ],
    )
    assert_test(test_case_1, [relevancy_metric])
    
if __name__ == "__main__":
    test_relevancy()