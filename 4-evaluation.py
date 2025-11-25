#!/usr/bin/env python
# coding: utf-8

# 

# # Evaluate with ROUGE

# In[1]:


# Define RetrievalQA chain
import utils  # Import the whole module so you can reload it
import importlib

importlib.reload(utils)

from utils import init_qa_chain
qa_chain = init_qa_chain()
print("✅ RetrievalQA chain created")


# In[2]:


import evaluate
from nltk.tokenize import sent_tokenize
import nltk

# Ensure required tokenizer
nltk.download('punkt')

# Load ROUGE metric
rouge = evaluate.load("rouge")

def compute_rouge(generated_text, reference_text):
    """
    Compute ROUGE score between a reference answer and a generated answer.
    """
    # Ensure text is split into sentences with newlines
    generated = "\n".join(sent_tokenize(generated_text.strip()))
    reference = "\n".join(sent_tokenize(reference_text.strip()))

    result = rouge.compute(
        predictions=[generated],
        references=[reference],
        use_stemmer=True,
    )
    return result


# In[3]:


def answer_with_rouge(input_text: str, reference_answer: str = None):
    print(f"\n📝 Query: {input_text}")
    result = qa_chain({"query": input_text})
    answer = result["result"]
    sources = list(set(doc.metadata.get("source_file", "Unknown") for doc in result["source_documents"]))
    print(f"🗒️ Retrieved {len(sources)} source documents")

    output = f"{answer}\n\nSources:\n" + "\n".join(sources)

    if reference_answer:
        rouge_scores = compute_rouge(answer, reference_answer)
        print(f"🔴 ROUGE scores:")
        for k, v in rouge_scores.items():
            print(f"  {k.upper()}: {v:.4f}")
        output += "\n\nROUGE scores:\n" + "\n".join(f"{k.upper()}: {v:.4f}" for k, v in rouge_scores.items())

    return output


# ### Testing with references

# In[4]:


query = "What is at the center of a black hole?"

reference = "The singularity at the center of a black hole is the ultimate no man's land: a place where matter is compressed down to an infinitely tiny point"
response = answer_with_rouge(query, reference)
print(response)


# In[8]:


query = "What is Bill Nye Famous for?"

reference = "He is best known as the host of the science education television show Bill Nye the Science Guy (1993–1999) and as a science educator in pop culture"
response = answer_with_rouge(query, reference)
print(response)


# In[9]:


query = "Who is Sara Imari Walker?"

reference = "Sara Imari Walker is an American theoretical physicist and astrobiologist with research interests in the origins of life, astrobiology, physics of life, emergence, complex and dynamical systems, and artificial life"
response = answer_with_rouge(query, reference)
print(response)


# In[10]:


query = "What is Sara Imari Walker known for?"

reference = "Sara Imari Walker is recognized for developing assembly theory, an informational framework for identifying life based on its complexity. "
response = answer_with_rouge(query, reference)
print(response)

