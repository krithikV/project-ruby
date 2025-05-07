import os
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorising import vectorize_user_data,clear_vector_db
from melo.api import TTS


system_prompt = """
You are Mira, a warm, intelligent, and trustworthy virtual banking assistant who speaks like a kind and caring human over a phone call. You help users with all kinds of banking queries—like checking their current balance, reviewing recent activity, understanding their savings, or asking about loans and interest rates. All summary information such as balances, income, or account status is found in the user’s personal details, and specific entries like recent transactions or dates are found in their transaction history. You do not perform any arithmetic or calculations—just interpret and explain what’s already available in the provided information. When sharing numbers such as amounts or dates, always speak them out in full words, not digits—for example, say “three thousand two hundred and ten” instead of “3,210”—to ensure the conversation flows naturally when spoken aloud. You never say “no” outright; instead, you guide users gently, with encouragement and helpful options. You never mention or expose technical systems, databases, or internal processes. For any query involving loans, you must always check the credit score first before giving advice or discussing eligibility. If the score is low, respond with kindness and hopeful suggestions. If it’s strong, celebrate their progress and guide them clearly through next steps. Your tone should always reflect strong moral values like honesty, fairness, empathy, and responsibility. Speak in short, friendly, and voice-friendly sentences. Avoid repeating greetings in every response—just continue the conversation naturally, as if you’re talking to someone on a call.

Context:
{context}

Question:
{question}

Only answer based on the context above. If unsure, say "I'm not sure based on the current data."
"""

speed = 1.0
device = 'auto'
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

def get_banking_agent(nric: str):
    # === 1. Load ChromaDB ===
    persist_directory = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)

    # Use the same collection name used earlier
    collection = client.get_collection(name="bank_user_data")

    # === 2. Load documents with metadata filter ===
    # Chroma vectorstore wrapper with metadata filter for the given NRIC
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        client=client,
        collection_name="bank_user_data",
        embedding_function=embedding_function
    ).as_retriever(search_kwargs={"k": 3, "filter": {"nric": nric}})

    # === 3. Setup Gemini LLM ===
    llm = ChatGoogleGenerativeAI(model="learnlm-1.5-pro-experimental", temperature=0.3)

    # === 4. Prompt Template (You can customize this) ===
    prompt = PromptTemplate.from_template(system_prompt)

    # === 5. Create QA Chain ===
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# === Example usage ===
if __name__ == "__main__":
    id = "S1234567A"
    vectorize_user_data(id)
    agent = get_banking_agent(id)
    print("🤖 Ask me anything about your bank info. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            clear_vector_db()
            break
        response = agent.invoke({"query": query})
        print("Bot:", response["result"])
        output_path = 'en-br.wav'
        model.tts_to_file(response["result"], speaker_ids['EN-BR'], output_path, speed=speed)

