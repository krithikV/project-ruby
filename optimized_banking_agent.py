import os
import shutil
import tempfile
import pandas as pd
import chromadb
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from melo.api import TTS

@dataclass
class BankingAgentConfig:
    """Configuration for the banking agent"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "bank_user_data"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1"
    llm_base_url: str = "http://213.181.123.59:40448/"
    llm_api_key: str = "e22545aaed47de5eac5e8306b45ae5e57a2b877762096ebbc5f5429a4d208a95"
    llm_temperature: float = 0.1
    tts_speed: float = 1.0
    tts_device: str = "auto"
    tts_language: str = "EN"
    melo_weights_path: str = "./melo_weights"  # Path to store pre-downloaded weights

class BankingAgent:
    """Optimized banking agent that combines vectorization and chat functionality"""
    
    def __init__(self, config: Optional[BankingAgentConfig] = None):
        self.config = config or BankingAgentConfig()
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize TTS, embeddings, and other components"""
        # Create and set permissions for directories
        for directory in [self.config.persist_directory, self.config.melo_weights_path]:
            if os.path.exists(directory):
                # Change permissions of existing directory and its contents
                for root, dirs, files in os.walk(directory):
                    os.chmod(root, 0o777)
                    for file in files:
                        os.chmod(os.path.join(root, file), 0o777)
            else:
                # Create new directory with proper permissions
                os.makedirs(directory, mode=0o777, exist_ok=True)
        
        # Initialize TTS
        self.tts_model = TTS(
            language=self.config.tts_language,
            device=self.config.tts_device
        )
        self.speaker_ids = self.tts_model.hps.data.spk2id
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.llm_base_url,
            headers={"Authorization": f"Bearer {self.config.llm_api_key}"}
        )
        
        # Initialize LLM
        self.llm = Ollama(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            headers={"Authorization": f"Bearer {self.config.llm_api_key}"},
            temperature=self.config.llm_temperature
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Load prompt template
        self.prompt_template = PromptTemplate.from_template("""
You are Mira, a warm, honest, and empathetic virtual banking assistant who answers only the user's direct question—nothing more. 
Use short, voice-friendly sentences. Read amounts and dates aloud in words ("three thousand two hundred and ten," "March first, two thousand twenty-five"). 
Never perform new calculations or mention internal systems. Before any loan discussion, check the user's credit score: 
if it's low, offer gentle encouragement; if strong, celebrate and explain next steps. Never say "no" outright—always offer a helpful alternative. 
Do not use emojis.

Chat History:
{chat_history}

Context:
{context}

Current Question:
{question}

Only answer based on the context above and chat history. If unsure, say "I'm not sure based on the current data."
""")

    def _clear_vector_db(self):
        """Clear the vector database"""
        if os.path.exists(self.config.persist_directory):
            shutil.rmtree(self.config.persist_directory)
            print("🧹 Cleared existing ChromaDB.")

    def _prepare_user_data(self, nric: str) -> Tuple[List[Dict], str]:
        """Prepare user data for vectorization"""
        # Load CSVs
        personal_df = pd.read_csv("data/personal_info.csv")
        loan_df = pd.read_csv("data/loan_history.csv")
        txn_df = pd.read_csv("data/transactions.csv")
        
        # Find user by NRIC
        user_info = personal_df[personal_df['nric'] == nric]
        if user_info.empty:
            raise ValueError(f"No user found with NRIC: {nric}")
        
        user_info = user_info.iloc[0]
        account_number = user_info["account_number"]
        
        # Prepare data chunks
        chunks = []
        
        # Personal info chunk
        personal_chunk = (
            f"User NRIC: {user_info.nric}, Name: {user_info.name}, Age: {user_info.age}, "
            f"Income: {user_info.income}, Credit Score: {user_info.credit_score}, "
            f"Bank Balance: {user_info.bank_balance}, Mobile: {user_info.mobile_number}, "
            f"Email: {user_info.email}"
        )
        chunks.append({"text": personal_chunk, "type": "personal_info"})
        
        # Loan history chunks
        user_loans = loan_df[loan_df['account_number'] == account_number]
        for _, row in user_loans.iterrows():
            loan_text = (
                f"Loan Type: {row.loan_type}, Amount: {row.loan_amount}, Status: {row.status}, "
                f"EMI: {row.monthly_emi}, Start Date: {row.start_date}"
            )
            chunks.append({"text": loan_text, "type": "loan"})
        
        # Transactions summary
        user_txns = txn_df[txn_df['account_number'] == account_number]
        txn_summary = (
            f"In the last month, there were {len(user_txns)} transactions. "
            f"Total spent: ${user_txns[user_txns['type'] == 'debit']['amount'].abs().sum():.2f}"
        )
        chunks.append({"text": txn_summary, "type": "transactions"})
        
        return chunks, account_number

    def initialize_agent(self, nric: str) -> None:
        """Initialize the banking agent for a specific user"""
        try:
            # Clear existing vector DB first
            self._clear_vector_db()
            
            # Initialize fresh ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.persist_directory
            )
            
            # Prepare user data
            chunks, _ = self._prepare_user_data(nric)
            
            # Create collection and add documents
            collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name
            )
            
            # Add documents to collection
            for idx, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk['text'])
                collection.add(
                    documents=[chunk['text']],
                    embeddings=[embedding],
                    metadatas=[{"nric": nric, "type": chunk["type"]}],
                    ids=[f"user-{nric}-chunk-{idx}"]
                )
            
            # Create vector store with filtered retriever
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings
            ).as_retriever(
                search_kwargs={"k": 3, "filter": {"nric": nric}}
            )
            
            # Create conversational chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": self.prompt_template}
            )
            
            print(f"✅ Banking agent initialized for user {nric}")
            
        except Exception as e:
            print(f"❌ Error initializing agent: {str(e)}")
            raise

    def get_response(self, question: str) -> Dict:
        """Get a response from the banking agent"""
        if not hasattr(self, 'qa_chain'):
            raise ValueError("Agent not initialized. Call initialize_agent() first.")
        
        response = self.qa_chain({"question": question})
        return response

    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech and return audio data"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
            self.tts_model.tts_to_file(
                text,
                self.speaker_ids['EN-BR'],
                output_path,
                speed=self.config.tts_speed
            )
            
            with open(output_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            os.unlink(output_path)
            return audio_data

# Example usage
if __name__ == "__main__":
    # Create agent instance
    agent = BankingAgent()
    
    # Initialize for a specific user
    nric = "S1234567A"
    agent.initialize_agent(nric)
    
    # Example interaction
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        try:
            response = agent.get_response(question)
            print("Mira:", response["answer"])
            
            # Convert to speech (optional)
            audio_data = agent.text_to_speech(response["answer"])
            # Save or play audio as needed
            
        except Exception as e:
            print(f"Error: {str(e)}") 