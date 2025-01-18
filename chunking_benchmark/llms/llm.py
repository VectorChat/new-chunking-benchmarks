FICTION_PROMPT = """
Bellow is a question that was created from a book, a list of answer choices, and some context that might be relevant to the question.
Your job is to select the best answer choice based on the question and context using only the context provided.
You should not use any other context or training data to answer the question, only use the context provided.
        
<QUESTION>
{question}

{choices_str}
</QUESTION>

<CONTEXT>
{context}   
</CONTEXT>

Relevant context will be provided in the CONTEXT section, do not use any other context. Even if the context is not relevant, you should still answer the question to the best of your ability.

Please think about your answer step by step before choosing an answer choice. After reasoning through your answer, provide your response in the following XML format:
<RESPONSE>
<REASONING>
[your reasoning for your answer choice]
</REASONING>
<ANSWER>
[your answer choice letter]
</ANSWER>
</RESPONSE>

Your answer should be a single letter (A, B, C, D, E, F, G, or H) and your reasoning should be a short paragraph explaining what context you used to come up with your answer and the steps you took
to reason about the question along with the answer choice you selected. Do not include brackets in your reasoning or answer.
"""
from langchain_openai import ChatOpenAI
from typing import Any

from schema import FictionAnswer
from dotenv import load_dotenv

load_dotenv()
class LLM:
    def __init__(
        self,
        model: Any = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=42),
        class_output: Any = FictionAnswer,
        prompt: str = FICTION_PROMPT,
        **kwargs,
    ):
        self.model = model
        self.structured_llm = self.model.with_structured_output(class_output)
        self.prompt = prompt

    def invoke_fiction(
        self,
        question: str,
        context: str,
        choices_str: str,
    ):
        prompt_ = self.prompt.format(
            question=question,
            context=context,
            choices_str=choices_str,
        )
        return self.invoke(prompt_)

    def invoke(self, prompt: str) -> Any:
        return self.structured_llm.invoke(prompt)
