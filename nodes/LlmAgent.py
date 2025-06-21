from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from typing import TypedDict, Optional
from pydantic import BaseModel, Field

class SummmarizeTextInput(BaseModel):
    text: str = Field(..., description="The text to summarize.")

class SummmarizeTextOutput(BaseModel):
    summary: str = Field(..., description="The summarized text.")

class ChatAzure:
    def __init__(self):
        self.llm =  AzureChatOpenAI(
            deployment_name="gpt-4.1",
            model="gpt-4.1",
            api_key=st.secrets["AZURE_OPENAI_API_KEY"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
            temperature=0.5,
            max_tokens=20000
        )

    def summarize_text(self, input: str) -> str:
        try:
            sumarizer_prompt_template = """You are a helpful assistant that summarizes text.
            Your task is to provide a concise summary of the following text:
            {text}
            Please provide the summary so that it captures the main points and essence of the text without losing important details.
            The summary should be clear, and easy to understand.
            You should response in Bahasa Indonesia."""
            prompt = PromptTemplate(
                input_variables=["text"],
                template=sumarizer_prompt_template
            )
            agent = prompt | self.llm.with_structured_output(SummmarizeTextOutput)
            response = agent.invoke({"text": input})
            if not response or not isinstance(response, SummmarizeTextOutput):
                return None
            return response.summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return None
