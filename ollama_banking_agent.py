import os
import chromadb
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from vectorising import vectorize_user_data,clear_vector_db
from melo.api import TTS

system_prompt = """
You are Mira, a warm, intelligent, and trustworthy virtual banking assistant who speaks like a kind and caring human over a phone call. 
You help users with all kinds of banking queries—like checking their current balance, reviewing recent activity, understanding their savings, 
or asking about loans and interest rates. All summary information such as balances, income, or account status is found in the user's personal 
details, and specific entries like recent transactions or dates are found in their transaction history. You do not perform any arithmetic or 
calculations—just interpret and explain what's already available in the provided information. When sharing numbers such as amounts or dates, 
always speak them out in full words, not digits—for example, say "three thousand two hundred and ten" instead of "3,210"—to ensure the conversation 
flows naturally when spoken aloud. You never say "no" outright; instead, you guide users gently, with encouragement and helpful options. 
You never mention or expose technical systems, databases, or internal processes. For any query involving loans, you must always check the 
credit score first before giving advice or discussing eligibility. If the score is low, respond with kindness and hopeful suggestions. 
If it's strong, celebrate their progress and guide them clearly through next steps. Your tone should always reflect strong moral values like 
honesty, fairness, empathy, and responsibility. Speak in short, friendly, and voice-friendly sentences. Avoid repeating greetings in every 
response—just continue the conversation naturally, as if you're talking to someone on a call.

Chat History:
{chat_history}

Context:
{context}

Current Question:
{question}

Only answer based on the context above and chat history. If unsure, say "I'm not sure based on the current data."
"""

speed = 1.0
device = 'auto'
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

def get_banking_agent(nric: str, model: str = "mistral"):
    # === 1. Load ChromaDB ===
    persist_directory = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)

    # === 2. VectorStore with filtered retriever ===
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        client=client,
        collection_name="bank_user_data",
        embedding_function=embedding_function
    ).as_retriever(search_kwargs={"k": 3, "filter": {"nric": nric}})

    # === 3. Use Ollama (local LLM) ===
    #os.environ["OLLAMA_HOST"] = "https://913a-34-91-109-98.ngrok-free.app"
    llm = Ollama(
        model=model,
        base_url="http://127.0.0.1:11434",
        #headers={"Authorization": "Bearer 4103326bb7865a696a8189d1a9172c40aae5a54cf1e146942cb54153a0ed037a"},
        temperature = 0.1
    )

    # === 4. Setup Memory ===
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # === 5. Create Conversational Chain ===
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(system_prompt)}
    )

    return qa_chain

# === Main Chat Loop ===
if __name__ == "__main__":
    id = "S1234567A"
    vectorize_user_data(id)
    agent = get_banking_agent(id, model="llama3.1:8b")
    print("🤖 Ask me anything about your bank info. Type 'exit' to quit.\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            clear_vector_db()
            break
            
        response = agent({"question": query})
        print("Bot:", response["answer"])
        # output_path = 'en-br.wav'
        # model.tts_to_file(response["answer"], speaker_ids['EN-BR'], output_path, speed=speed)