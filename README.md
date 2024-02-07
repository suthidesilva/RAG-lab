# XML Document Analysis with Conversational Agent

This project enables XML document analysis, particularly for product reviews, using a conversational agent. The agent is capable of processing user queries and retrieving relevant information from a vector store constructed from XML data.

## Installation

To use this code, you need to install the required dependencies:
**!pip install numexpr openai langchain_openai langchain langchain-community duckduckgo-search loguru qdrant-client sentence-transformers tiktoken docx2txt unstructured tqdm**

Additionally, ensure you have access to the necessary libraries and resources. For Google Colab users, mounting Google Drive is necessary for accessing XML review files.

## Usage
Parsing XML Documents: The parse_xml_to_dict function parses XML files containing product reviews into a structured format.
Building Vector Store: Use build_vectorstore_from_xml_directory to construct a vector store from XML documents. This store facilitates similarity searches based on user queries.
Setting Up Conversational Agent: Initialize a conversational agent with initialize_agent. This agent interacts with users, processes queries, and retrieves relevant information from the vector store.
