from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate

with open("faqs.txt", "r", encoding="utf-8") as f:
    faq_text = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=0
)

documents = text_splitter.create_documents([faq_text])

embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(documents, embeddings)

llm = Ollama(model="llama3.2")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an FAQ answering assistant.
Answer ONLY from the provided context.
If the answer is not present, say:
"I don't have that information."

Context:
{context}

Question:
{question}

Answer:
"""
)
print("\n\n\n--------------------------------------------------------\n")
print("FAQ Agent is ready. Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query, k=1)

    if not docs:
        print("Agent: I don't have that information.")
        continue

    context = docs[0].page_content

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)
    print("Agent:", response)
