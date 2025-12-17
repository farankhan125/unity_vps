import os
import numpy as np
from dotenv import load_dotenv
import time
import random
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_astradb import AstraDBVectorStore
from questions import questions
from embedded_faqs import embedded_faqs

# Functions Below

def pick_random_questions(questions, top=3):
    """
    Returns `top` number of random questions from the list.
    
    :param questions: list of questions
    :param top: how many questions to return
    :return: list of randomly selected questions
    """
    # Ensure we don't request more questions than available
    top = min(top, len(questions))
    
    return random.sample(questions, top)

def get_top_faqs(query, embedded_faqs, embedding_model, top_k=2):
    """
    Finds the top_k most similar FAQs to a query using cosine similarity.

    Parameters
    ----------
    embedded_faqs : list[dict]
        Each dict must have keys: "faq" and "embedding" (list of floats).
    
    query : str
        The user question to search for.
    
    embedding_model : OpenAIEmbeddings
        LangChain embeddings instance to embed the query.
    
    top_k : int, optional
        Number of top similar FAQs to return (default is 3).

    Returns
    -------
    list[dict]
        Top_k FAQs sorted by similarity, each dict has keys: "faq" and "similarity".
    """
    # Embed the query
    query_embedding = embedding_model.embed_query(query)

    # Compute cosine similarity with each FAQ
    faqs_with_similarity = []
    for item in embedded_faqs:
        faq_embedding = np.array(item["embedding"])
        query_vec = np.array(query_embedding)
        
        # cosine similarity formula
        similarity = np.dot(query_vec, faq_embedding) / (np.linalg.norm(query_vec) * np.linalg.norm(faq_embedding))
        
        faqs_with_similarity.append({
            "faq": item["faq"],
            "similarity": similarity
        })

    # Sort by similarity descending
    faqs_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)

    # Take top_k results
    top_faqs = faqs_with_similarity[:top_k]

    # Build a clean, formatted string
    formatted_faqs = []
    for i, item in enumerate(top_faqs, 1):
        formatted_faqs.append(f"FAQ {i}:\n{item['faq']}")

    # Return as one string
    return "\n\n".join(formatted_faqs)

# Functions Above

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)

# Reload vector DB (no re-embedding, fast)
vector_store = AstraDBVectorStore(
        collection_name="UIS_Data",       
        embedding=embedding_model,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],       
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],           
        namespace=None         
)

contextualize_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = """
You are the Unity In Spirit assistant chatbot, helping users with questions about Unity In Spirit and its services with creativity, clarity, and confidence. 
Always respond based on the provided context, Related FAQS and focus only on what Unity In Spirit offers — explain, recommend, or guide users toward relevant Unity In Spirit services, not general ideas or advice. 
Keep your responses clear, friendly, and professional. Be concise but complete, ensuring the user understands how Unity In Spirit can help. 
Always respond based on the user query to the point.
If the question is outside Unity In Spirit services or unrelated to what Unity In Spirit provides, politely respond: 
"I am here to assist with Unity In Spirit services only."
Do not provide unrelated or speculative ideas.

# Retrieved Knowledge (RAG)
Use this context only if relevant:
{context}

# Related FAQs (Optional)
These FAQs might help but DO NOT override factual context:
{faqs}
"""

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

retriever = vector_store.as_retriever(search_kwargs={'k': 3})
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    document_chain
)

chat_histories: Dict[str, List] = {}
session_timestamps = {}

# Routes

class UserInput(BaseModel):
    user_input: str
    session_id: str

@app.post("/ai-answer")
def generate_answer(request: UserInput):

    # Clean expired sessions (10 min)
    for sid in list(session_timestamps.keys()):
        if time.time() - session_timestamps[sid] > 600:
            chat_histories.pop(sid, None)
            session_timestamps.pop(sid, None)
    
    session_id = request.session_id

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    session_timestamps[session_id] = time.time()

    try:
        # Get top relevant FAQs for this query
        top_faqs_str = get_top_faqs(
            query=request.user_input,
            embedded_faqs=embedded_faqs,
            embedding_model=embedding_model,
            top_k=2
        )

        # Get RAG-based AI answer
        response = rag_chain.invoke({
            "chat_history": chat_histories[session_id],
            "input": request.user_input,
            "faqs": top_faqs_str,
        })

        ai_answer = response["answer"]

        # Add conversation to chat history
        chat_histories[session_id].extend([
            HumanMessage(content=request.user_input),
            AIMessage(content=ai_answer)
        ])

        # Get top 3 relevant questions
        top_questions = pick_random_questions(questions, top=3)

        return {
            "answer": ai_answer,
            "related_questions": top_questions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Below is just for testing and running server locally (remove below when uploading it on a cloud)
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))

#     uvicorn.run("main:app", host="0.0.0.0", port=port)
