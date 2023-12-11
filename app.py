import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.schema.language_model import BaseLanguageModel
import gradio as gr

# Load environment variables from .env file
load_dotenv(override=True)

# Set up MongoDB connection details
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "pdfchatbot"
COLLECTION_NAME = "pdfText"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Initialize OpenAIEmbeddings with the API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define field names
EMBEDDING_FIELD_NAME = "embedding"
TEXT_FIELD_NAME = "text"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def get_embedding(text):
    text = text.replace("\n", " ")
    # return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    client = OpenAI()

    response = client.embeddings.create(input=text, model="text-embedding-ada-002")

    return response.data[0].embedding


def find_similar_documents(embedding):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    documents = list(
        collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 20,
                        "limit": 10,
                    }
                },
                {"$project": {"_id": 0, "text": 1}},
            ]
        )
    )
    return documents


# Function to process PDF document


def process_pdf(file):
    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()

    # Print loaded pages
    print(pages)

    # Split text into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    # Remove single spaces between letters and replace double spaces
    # docs_cleaned = [re.sub(r'\s+', ' ', doc.replace(' ', '')) for doc in docs]

    # Convert 'Document' objects to strings
    docs_as_strings = [str(doc) for doc in docs]

    # Set up MongoDBAtlasVectorSearch with embeddings
    vectorStore = MongoDBAtlasVectorSearch(
        collection, embeddings, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )

    # Insert the documents into MongoDB Atlas Vector Search
    docsearch = vectorStore.from_documents(
        docs,
        embeddings,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return docsearch


# Function to query and display results


def query_and_display(query):
    # Set up MongoDBAtlasVectorSearch with embeddings
    vectorStore = MongoDBAtlasVectorSearch(
        collection,
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    print(query)
    # Query MongoDB Atlas Vector Search
    # results = vectorStore.similarity_search(
    #     query=query,
    #     k=20,
    #     collection=collection,
    #     index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    # )
    print("---------------")
    docs = vectorStore.max_marginal_relevance_search(query, K=5)

    # embed = get_embedding(query)

    # results = find_similar_documents(embed)

    # Print the result from MongoDB Atlas Vector Search
    # results = ""
    # for document in docs:
    #     print(str(document) + "\n")
    #     results = results + str(document["text"]) + "\n\n"
    #     # docs = docs + str(document) + "\n\n"

    # Leveraging Atlas Vector Search paired with Langchain's QARetriever

    # Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
    # If it's not specified (for example like in the code below),
    # then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    # compressor = LLMChainExtractor.from_llm(llm)

    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=vectorStore.as_retriever()
    # )
    # print("\nAI Response:")
    # print("-----------")
    # compressed_docs = compression_retriever.get_relevant_documents(query)
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    # print(compressed_docs[0].metadata['title'])
    # print(retriever[0].page_content)
    for document in retriever:
        print(str(document) + "\n")


    qa = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever
    )
    retriever_output = qa.run(query)

    print(retriever_output)

    # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
    # Implements _get_relevant_documents which retrieves documents relevant to a query.
    # retriever = vectorStore.as_retriever()

    # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
    # inserts them all into a prompt and passes that prompt to an LLM.

    # qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Execute the chain

    # retriever_output = qa.run(query)
    # return retriever_output

    # Return Atlas Vector Search output, and output generated using RAG Architecture
    return retriever_output


with gr.Blocks() as demo:
    gr.Markdown("Powerful chatbot drop files or ask questions using this site.")

    with gr.Tab("Upload PDF"):
        with gr.Row():
            pdf_input = gr.File()
            pdf_output = gr.Textbox()
        pdf_button = gr.Button("Upload PDF")

    with gr.Tab("Ask question"):
        with gr.Row():
            question_input = gr.Textbox()
            question_output1 = gr.Textbox()
            # question_output2 = gr.Textbox()
        question_button = gr.Button("Ask question")

    pdf_button.click(process_pdf, inputs=pdf_input, outputs=pdf_output)
    question_button.click(
        query_and_display, inputs=question_input, outputs=question_output1
    )

demo.launch()
