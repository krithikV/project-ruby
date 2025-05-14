import streamlit as st
import os
import base64
from optimized_banking_agent import BankingAgent, BankingAgentConfig

def get_audio_player(audio_data, autoplay=False):
    """Create an audio player widget with the given audio data"""
    audio_base64 = base64.b64encode(audio_data).decode()
    autoplay_attr = "autoplay" if autoplay else ""
    audio_html = f"""
        <audio controls {autoplay_attr}>
            <source src=\"data:audio/wav;base64,{audio_base64}\" type=\"audio/wav\">
            Your browser does not support the audio element.
        </audio>
    """
    return audio_html

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'nric' not in st.session_state:
        st.session_state.nric = None

def main():
    st.title("Mira - Your Banking Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for NRIC input
    with st.sidebar:
        st.header("Login")
        nric = st.text_input("Enter your NRIC (e.g., S1234567A)", 
                           value=st.session_state.nric if st.session_state.nric else "")
        
        if nric and nric != st.session_state.nric:
            st.session_state.nric = nric
            # Initialize agent with new NRIC
            try:
                # Create and initialize banking agent
                config = BankingAgentConfig()  # Use default config
                st.session_state.agent = BankingAgent(config)
                st.session_state.agent.initialize_agent(nric)
                st.success("Successfully logged in!")
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
                st.session_state.agent = None
    
    # Main chat interface
    if st.session_state.agent is None:
        st.info("Please enter your NRIC in the sidebar to start chatting with Mira.")
    else:
        # Display chat messages
        num_msgs = len(st.session_state.messages)
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("audio"):
                    # Only autoplay for the latest assistant message
                    autoplay = (
                        message["role"] == "assistant" and idx == num_msgs - 1
                    )
                    st.markdown(get_audio_player(message["audio"], autoplay=autoplay), unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask Mira about your banking information"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.get_response(prompt)
                    answer = response["answer"]
                    
                    # Convert response to speech using the agent's TTS
                    audio_data = st.session_state.agent.text_to_speech(answer)
                    
                    # Display response and audio
                    st.write(answer)
                    st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "audio": audio_data
                    })
        
        # Add a clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 