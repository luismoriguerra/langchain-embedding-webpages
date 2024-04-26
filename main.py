
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings



model_local = ChatOllama(model='mistral')

urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=7500, chunk_overlap=100, add_start_index=True
)
doc_splits = text_splitter.split_documents(docs_list)


vecorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)

retriever = vecorstore.as_retriever()

# 3. before rag
print("before rag\n")
before_rag_template ="what is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Ollama"}))


# 4. after rag
print("##########\n")
print("after rag\n")
after_rag_template= """Answer the question based only on the following context:
{context}
Question: {question}
"""

after_rag_prompt= ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

print(after_rag_chain.invoke("What is Ollama?"))