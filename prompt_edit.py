import langchain
import sys
import os

from langchain.prompts.prompt import PromptTemplate

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Preserve the ogirinal question in the answer setiment during rephrasing.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT_CUSTOM = PromptTemplate.from_template(custom_template)