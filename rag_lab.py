# -*- coding: utf-8 -*-
"""RAG lab.ipynb

Suthi de Silva
"""

! pip install numexpr openai langchain_openai langchain langchain-community duckduckgo-search loguru qdrant-client sentence-transformers tiktoken docx2txt unstructured tqdm

from typing import List, Dict, Any

import xml.etree.ElementTree as ET
import json

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.tools.retriever import create_retriever_tool


def parse_xml_to_dict(file_path: str) -> Dict[str, Any]:
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Function to parse each review element
    def parse_review(review_element: ET.Element) -> Dict[str, Any]:
        review_data = {}
        for child in review_element:
            if child.tag == 'reviewer':
                # Special handling for nested 'reviewer' tag
                reviewer_data = {grandchild.tag: grandchild.text for grandchild in child}
                review_data[child.tag] = reviewer_data
            else:
                review_data[child.tag] = child.text
        return review_data

    # Parse all reviews
    reviews = [parse_review(review) for review in root]
    return reviews


class XMLReviewLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        docs = parse_xml_to_dict(self.file_path)
        return [Document(page_content=json.dumps(doc)) for doc in docs]


class VerboseEmbeddings(Embeddings):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"Embedding {len(texts)} documents")
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        print(f"Embedding query: {text}")
        return self.embeddings.embed_query(text)


def build_vectorstore_from_xml_directory(path: str, vectorstore_path: str, loading_glob="**/*.xml",
                                         collection_name="my_documents",
                                         embeddings: Embeddings = VerboseEmbeddings(HuggingFaceEmbeddings())):
    # setting up a document loader which loads all xml reviews in the directory
    dir_loader = DirectoryLoader(path=path, glob=loading_glob, loader_cls=XMLReviewLoader,
                                 show_progress=True, use_multithreading=True)
    docs = dir_loader.load()
    # splitting the xml documents into chunks of 1000 characters with an overlap of 200 characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=100, add_start_index=True
    )
    all_splits: List[Document] = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(all_splits)} chunks")
    # setting up a vectorstore with the Qdrant backend
    vectorstore = Qdrant.from_documents(
        all_splits,
        embeddings,
        path=vectorstore_path,
        collection_name=collection_name
    )
    # search args could return a different number of nearest neighbors.  Arbitrarily putting "6" here.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever


def retriever_tool(retriever):
    """Creates a tool that searches the vectorstore for documents similar to the input text."""
    tool = create_retriever_tool(
        retriever,
        "search_xml_documents",
        "Searches inside of our local store old Best Buy reviews.  " +
        "Useful for understanding how people talk about products from the early 2010's era",
    )
    return tool

from google.colab import drive
drive.mount('/content/drive')

!ls '/content/drive/MyDrive/product_reviews/'

review_dir = "/content/drive/MyDrive/product_reviews/"

retriever = build_vectorstore_from_xml_directory(collection_name="test", path=review_dir, loading_glob="**/reviews_0001_24122_to_98772.xml", vectorstore_path='/content/reviews')

review_tool = retriever_tool(retriever)

!pip install langchain

import os
from getpass import getpass
from langchain.agents import AgentExecutor, AgentType, load_tools, initialize_agent
from langchain.chat_models import ChatAnyscale
from langchain_openai import OpenAIEmbeddings
# from langchain.memories import ConversationalBufferSummaryMemory, VectorRetrieverMemory # Initialize Memory Components

from google.colab import userdata

llm = ChatAnyscale(anyscale_api_base="https://api.endpoints.anyscale.com/v1",
                   anyscale_api_key=userdata.get("ANYSCALE_API_KEY"),
                   model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                   temperature=0.7,
                   verbose=True)

tools = load_tools(["llm-math", "ddg-search"], llm=llm) + [review_tool]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.agent.llm_chain.prompt.template)



# # Define your own prompt and questions
# prompt_template = '''
# Question: {input}
# '''

# questions = [
#     "In 2015, what Bluetooth headset was the most popular?",
#     "Tell me about the top Bluetooth headsets in 2015.",
#     "Which Bluetooth headset had the highest sales in 2015?",
#     # Add more questions related to Bluetooth headsets in 2015
# ]

# # Initialize the AgentExecutor chain and loop through the questions
# for question in questions:
#     print(f"\nQuestion: {question}")

#     # Prepare the prompt for the question
#     prompt = prompt_template.format(input=question)

#     # Execute the chain for the given question
#     # Replace 'execute' with the correct method name if different
#     response = agent.execute(
#         prompt=prompt,
#         agent=agent.agent
#     )

#     # Print the agent's response
#     print(f"Final Answer: {response['output']}")

# # End of loop
# print("\nAll questions have been answered.")

# def refine_search_query(original_query: str) -> str:
#     # Add specific keywords or patterns to refine the query
#     refined_query = f"{original_query} bestseller ranking"
#     return refined_query

# def analyze_search_results(results: List[str]) -> str:
#     # Analyze the results to find the most relevant information
#     # This is a placeholder implementation, adjust according to your actual results
#     for result in results:
#         if "2015" in result and "Bluetooth headset" in result:
#             return result
#     return "Specific information not found."

# def fallback_strategy():
#     return "I couldn't find specific information. You might want to try searching with different keywords or checking specialized historical data sources."

# def interactive_feedback(user_query: str):
#     # Interactive feedback mechanism
#     # Ask user to refine query or provide more context
#     refined_query = input("Could you provide more specific keywords or context? ")
#     return refined_query if refined_query else user_query

# # Example usage in your agent's execution flow
# user_query = "In 2015, which Bluetooth headset was the most popular?"
# refined_query = refine_search_query(user_query)

# # Assuming 'perform_search' is a function that uses duckduckgo_search or similar
# search_results = perform_search(refined_query)
# analyzed_result = analyze_search_results(search_results)

# if analyzed_result == "Specific information not found.":
#     user_query = interactive_feedback(user_query)
#     search_results = perform_search(user_query)
#     analyzed_result = analyze_search_results(search_results)
#     if analyzed_result == "Specific information not found.":
#         analyzed_result = fallback_strategy()

# print(analyzed_result)

"""I aim to enhance your conversational agent's capabilities, particularly in improving the effectiveness of handling search queries. To achieve this, I've incorporated additional steps into the query processing flow, focusing on refining the search, analyzing results, and facilitating user interaction for more precise input.

**Refine the Search Query (refine_search_query function):**
This function is designed to take an original query and enhance it by incorporating specific keywords or patterns. In this instance, it augments the original query with "bestseller ranking" to enhance search results, striving to make the query more specific.

**Analyze Search Results (analyze_search_results function):**
Following the execution of a search with the refined query, this function scrutinizes the results. It specifically seeks out outcomes that explicitly mention "2015" and "Bluetooth headset," indicating relevance to the user's query. If such results are identified, they are returned as the output.

**Fallback Strategy (fallback_strategy function):**
In cases where the search results fail to provide specific or pertinent information, this function steps in with a fallback response.

**Analyze Search Results (analyze_search_results function):**
This function comes into play when the initial search results are unsatisfactory. It prompts the user to refine their query or offer more specific context, encouraging user interaction to better understand their information needs.

These enhancements are geared towards elevating the conversational agent's proficiency in managing search queries, encompassing refined query processing, intelligent result analysis, and responsive fallback strategies.
"""

agent.invoke("In 2015 was bluetooth head set was the most popular")

agent.invoke("In 2015 what bluetooth headset was the most popular")

"""For Lab 5, I embarked on extensive exploration and form analysis. Although I wrote some code, it didn't function as intended.

My strategy involved initializing the memory component after importing the necessary libraries and before initializing the agent.

To achieve this, it is recommended to modify the agent initialization process by incorporating the memory components.

Additionally, updating the agent's execution flow is crucial. This entails adjusting how the agent processes input, ensuring it checks the memory components either before or during the execution of queries.

Given the intricacies of your existing setup, it is imperative to integrate these changes with caution.

I have commented out sections in the code, indicating my attempts to add them to the Colab. I will provide explanations in text cells underneath.

Furthermore, I experimented with questioning the agent and observed that precision and error-free spelling were essential. For instance, the question 'n 2015 was bluetooth head set was the most popular?' did not yield a satisfactory answer. On the other hand, a similar well-phrased question like 'In 2015 what bluetooth headset was the most popular?' produced a much more accurate response.


"""
