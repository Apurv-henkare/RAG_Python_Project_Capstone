from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_REASONING_EFFORT,
    LLM_MAX_TOKENS,
)
from schemas import parser


def get_llm() -> ChatOpenAI:
    """
    Initializes and returns the ChatOpenAI client configured for Gemini.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=GEMINI_API_KEY,
        base_url=LLM_BASE_URL,
        reasoning_effort=LLM_REASONING_EFFORT,
        max_tokens=LLM_MAX_TOKENS,
    )


def format_docs(docs: List[Document]) -> str:
    """
    Combines retrieved documents into a single text string for context injection.
    """
    return "\n\n".join(d.page_content for d in docs)


def get_prompt_template() -> PromptTemplate:
    """
    Builds the PromptTemplate with formatting instructions for structured output.
    """
    template_str = """You are an AI assistant helping users understand Love2D game development documentation.

Use ONLY the information provided in the context below to answer the question.

Guidelines:
- Stay faithful to the original documentation's terminology and code examples.
- When the user asks for code, provide COMPLETE, working code snippets with ```lua markdown code blocks. Never truncate or cut off code examples.
- In each answer item:
  * 'text': Provide the thorough explanation AND any complete code snippets requested.
  * 'context': Provide the supporting excerpt or code from the documentation IF AND ONLY IF it directly supports the answer.
- If the question is out of scope (e.g., asking about other programming languages like C#, Python, or topics not in Love2D) or the context is insufficient:
  * Explicitly state in 'text' that the documentation does not contain this information.
  * Set 'context' to an EMPTY string (""). NEVER include irrelevant excerpts or quotes when the question cannot be answered.
- If multiple sections support the answer, synthesize them clearly.

Context:
{context}

Question:
{question}

Answer (clear, complete, and well-structured):
When answering:
- follow the following schema: {format_instructions}
- Do not invent nonexistent APIs or methods.
- Make sure code snippets are fully written out and closed properly.
"""
    return PromptTemplate(
        template=template_str,
        input_variables=["question", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


def create_rag_chain(retriever):
    """
    Assembles the complete RAG LCEL pipeline:
    Query -> Parallel(question, retrieved formatted context) -> Prompt -> LLM -> Pydantic Output Parser
    """
    llm = get_llm()
    prompt = get_prompt_template()

    q_chain = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": retriever | RunnableLambda(format_docs),
    })

    rag_chain = q_chain | prompt | llm | parser
    return rag_chain
