import os
import pandas as pd
import argparse
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
import numpy as np
import nltk
import evaluate

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# --- Setup and Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")


def extract_claims_with_llm(text_to_evaluate: str, llm) -> set:
    """
    Uses a powerful LLM to extract a set of specific, short legal entities from a text.
    """
    if not text_to_evaluate or not isinstance(text_to_evaluate, str):
        return set()

    # --- NEW: More specific prompt for targeted entity extraction ---
    # This prompt instructs the LLM to pull out only short entities and provides examples.
    claim_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert legal entity and fact extractor. Your task is to extract ONLY the following from the given text:
1.  **Acts**: The full name of any mentioned law (e.g., 'Hindu Marriage Act, 1955').
2.  **Sections**: Any mentioned section numbers (e.g., 'Section 138').
3.  **Core Facts/Claims**: Very short, crucial facts, typically 2-4 words (e.g., 'grounds of cruelty', 'cheque dishonour').

DO NOT extract long sentences or explanations. Be concise. List each extracted item on a new line.

---
EXAMPLE 1
Text: 'The Hindu Marriage Act, 1955, under Section 13(1)(ia), allows for divorce on the grounds of cruelty. Cruelty is not limited to physical harm but can also encompass mental agony and suffering.'
Your Output:
Hindu Marriage Act, 1955
Section 13(1)(ia)
grounds of cruelty
mental agony

---
EXAMPLE 2
Text: 'Section 138 of the Negotiable Instruments Act, 1881, deals with the dishonour of a cheque for insufficiency of funds. A legal notice must be sent to the drawer within 30 days.'
Your Output:
Section 138
Negotiable Instruments Act, 1881
dishonour of a cheque
insufficiency of funds
legal notice within 30 days
---
"""),
        ("human", "Please extract the entities and facts from the following text:\n\n{text}")
    ])
    
    # The rest of the chain logic remains the same
    extractor_chain = claim_extraction_prompt | llm | StrOutputParser()
    
    try:
        response = extractor_chain.invoke({"text": text_to_evaluate})
        # Split response into lines and create a set of non-empty, stripped claims
        claims = {claim.strip().lower() for claim in response.split('\n') if claim.strip()}
        return claims
    except Exception as e:
        print(f"Error during claim extraction: {e}")
        return set()       




# --- Main Execution ---

if __name__ == "__main__":
    eval_llm = ChatGroq(temperature=0.4, model_name="llama-3.3-70b-versatile")
    generated_answer='''

Dear Client,A domicile certificate or a residential certificate is compulsory for MHT CET admission, especially for students seeking admission to Maharashtra State-based colleges. It's used to determine if a candidate is eligible for the Maharashtra state quota and any associated benefits. Any person from the State of Maharashtra could obtain this certificate, provided that he or she has been a resident in the State for the last 15 years. However, in a notification issued on May 13, 2013, the Director of Technical Education (DTE) has stated that the birth certificate or the school leaving certificate (SLC), which mentions the place of birth as in Maharashtra, will be accepted as an alternative to the domicile certificate to be submitted along with the admission form by the Maharashtra state candidates. For further clarification on the subject, you may contact the offices of the Sub-Divisional Magistrate, the Tehsildar’s office, and the District Collector’s office.



'''
    claims = extract_claims_with_llm(generated_answer, eval_llm)
    print(claims)



