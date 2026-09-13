from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Answer(TypedDict):
    text: str
    context: str


class Transformer(BaseModel):
    question: str = Field(description="The user's original query")
    answers: List[Answer] = Field(description="List of structured answers derived from context")


parser = PydanticOutputParser(pydantic_object=Transformer)
