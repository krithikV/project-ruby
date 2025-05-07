import os
import shutil
import pandas as pd
import chromadb
import requests

# --- Ollama embedding function ---
def get_ollama_embedding(text, model='nomic-embed-text'):
    url = 'http://localhost:11434/api/embeddings'
    payload = {
        'model': model,
        'prompt': text
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()['embedding']

def vectorize_user_data(nric: str):
    # === 1. Clear existing ChromaDB if it exists ===
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("🧹 Cleared existing ChromaDB.")

    # === 2. Initialize Chroma client (new API) ===
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # === 3. Load CSVs ===
    personal_df = pd.read_csv("data/personal_info.csv")
    loan_df = pd.read_csv("data/loan_history.csv")
    txn_df = pd.read_csv("data/transactions.csv")

    # === 4. Find user by NRIC ===
    user_info = personal_df[personal_df['nric'] == nric]
    if user_info.empty:
        raise ValueError(f"No user found with NRIC: {nric}")
    user_info = user_info.iloc[0]
    account_number = user_info["account_number"]

    # === 5. Create chunks for vectorization ===
    chunks = []

    # Personal info (updated with new fields)
    personal_chunk = (
        f"User NRIC: {user_info.nric}, Name: {user_info.name}, Age: {user_info.age}, "
        f"Income: {user_info.income}, Credit Score: {user_info.credit_score}, "
        f"Bank Balance: {user_info.bank_balance}, Mobile: {user_info.mobile_number}, "
        f"Email: {user_info.email}"
    )
    chunks.append({"text": personal_chunk, "type": "personal_info"})

    # Loan history
    user_loans = loan_df[loan_df['account_number'] == account_number]
    for _, row in user_loans.iterrows():
        loan_text = (
            f"Loan Type: {row.loan_type}, Amount: {row.loan_amount}, Status: {row.status}, "
            f"EMI: {row.monthly_emi}, Start Date: {row.start_date}"
        )
        chunks.append({"text": loan_text, "type": "loan"})

    # Transactions summary
    user_txns = txn_df[txn_df['account_number'] == account_number]
    txn_summary = f"In the last month, there were {len(user_txns)} transactions. "
    total_spent = user_txns[user_txns['type'] == 'debit']['amount'].abs().sum()
    txn_summary += f"Total spent: ${total_spent:.2f}"
    chunks.append({"text": txn_summary, "type": "transactions"})

    # === 6. Create collection and add documents ===
    collection = chroma_client.get_or_create_collection(name="bank_user_data")

    for idx, chunk in enumerate(chunks):
        embedding = get_ollama_embedding(chunk['text'])
        collection.add(
            documents=[chunk['text']],
            embeddings=[embedding],
            metadatas=[{"nric": nric, "type": chunk["type"]}],
            ids=[f"user-{nric}-chunk-{idx}"]
        )

    print(f"✅ Vector DB created for user {nric} with {len(chunks)} chunks.")

def clear_vector_db():
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("🧹 Cleared existing ChromaDB.")


# Example usage
#vectorize_user_data("S1234567A")
