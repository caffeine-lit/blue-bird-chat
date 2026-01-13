import streamlit as st
import google.generativeai as genai
from google.generativeai import caching
import os

MODEL = "gemini-3-pro-preview"

# --- 1. PAGE CONFIGURATION (Wide Mode) ---
st.set_page_config(
    page_title="Transcript Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1.5. CUSTOM STYLES (Make things bigger) ---
st.markdown("""
<style>
    /* Increase base font size for chat messages */
    .stChatMessage .stMarkdown p {
        font-size: 1.25rem !important;
        line-height: 1.6 !important;
    }
    /* Increase font size for input area */
    .stChatInput textarea {
        font-size: 1.15rem !important;
    }
    /* Increase expander summary text */
    .streamlit-expanderHeader {
        font-size: 1.15rem !important;
    }
    /* General body text increase */
    p {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. API SETUP ---
# Priority: 1. Environment Variable, 2. Session State (User Input)
api_key = os.getenv("AI_STUDIO_API_KEY") or st.session_state.get("api_key")

if not api_key:
    auth_placeholder = st.empty()
    with auth_placeholder.container():
        st.header("🔑 Authentication")
        st.warning("API Key not found in environment variables.")
        api_key_input = st.text_input("Enter Google API Key:", type="password", key="temp_api_key_input")
        if api_key_input:
            st.session_state.api_key = api_key_input
            auth_placeholder.empty()
            st.rerun()
        else:
            st.info("Please enter your Google API Key to proceed.")
            st.stop()

genai.configure(api_key=api_key)

# --- 3. UPLOAD TRANSCRIPTS ---
st.title("🤖 Intelligent Transcript Chat")

st.subheader("1. Upload Transcripts")
uploaded_files = st.file_uploader(
    "Upload one or more transcript files (.txt) to begin", 
    type=["txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👋 Welcome! Please upload transcript files above to start the chat.")
    st.stop()

# Combine content
combined_content = ""
file_names = []
for uploaded_file in uploaded_files:
    # Read and decode
    content = uploaded_file.getvalue().decode("utf-8")
    combined_content += f"\n\n--- SOURCE: {uploaded_file.name} ---\n\n"
    combined_content += content
    file_names.append(uploaded_file.name)

TRANSCRIPT = combined_content

# Manage History Reset on Content Change
current_content_hash = hash(TRANSCRIPT)
if "content_hash" not in st.session_state or st.session_state.content_hash != current_content_hash:
    st.session_state.messages = []
    st.session_state.content_hash = current_content_hash

# --- 4. CONTEXT CACHING ---
@st.cache_resource
def get_cached_content(transcript_text):
    """
    Creates a cache for the transcript. 
    Streamlit's cache_resource ensures this only runs once.
    """
    # Create the cache
    cache = caching.CachedContent.create(
        model=MODEL,
        display_name='transcript_cache', 
        system_instruction = (
            "You are a helpful analyst assistant whose knowledge is primarily grounded in the provided TRANSCRIPT. "
            "Adhere to the following guidelines when answering:\n"
            "1. **Priority**: Always attempt to answer using the transcript first.\n"
            "2. **Out of Scope**: If the user asks about a topic not found in the transcript, you must explicitly state that the topic is 'outside the scope of the lesson.'\n"
            "3. **General Knowledge**: After the disclaimer, provide accurate, factual information about the topic using your general knowledge.\n"
            "4. **Synthesis**: If applicable, draw parallels or comparisons between this outside information and concepts found in the transcript.\n"
            "Constraint: Maintain strict factual accuracy and do not hallucinate information."
        ),
        contents=[transcript_text]
    )
    return cache

try:
    with st.spinner("Processing and caching transcripts..."):
        cache = get_cached_content(TRANSCRIPT)
except Exception as e:
    st.error(f"Failed to create cache: {e}")
    st.stop()

# --- 5. SESSION STATE (History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. MAIN UI SETUP ---
st.divider()
token_count = cache.usage_metadata.total_token_count if cache else 0
st.caption(f"Active Context: {', '.join(file_names)} • {token_count:,} tokens")

with st.expander("📄 View Combined Transcript Source"):
    st.text(TRANSCRIPT)

# --- 7. RENDER HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "usage" in message:
            u = message["usage"]
            cached_info = f" [Cached: {u['cached_tokens']}]" if u.get("cached_tokens") else ""
            st.caption(f"Tokens: {u['total_tokens']} (Prompt: {u['prompt_tokens']}{cached_info}, Response: {u['candidates_tokens']})")

# --- 8. CHAT LOGIC ---
if prompt := st.chat_input("Ask a question about the transcripts..."):

    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # B. Add to local history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # C. Map history for Google API (User -> user, Assistant -> model)
    google_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in st.session_state.messages
    ]

    # D. Generate Response
    try:
        # Initialize Model from Cache
        model = genai.GenerativeModel.from_cached_content(cached_content=cache)

        # Start Chat Session (excluding the very last prompt which we send via send_message)
        chat = model.start_chat(history=google_history[:-1])
        
        # Send message
        with st.spinner("Thinking..."):
            response = chat.send_message(prompt)

        # E. Display AI Response
        with st.chat_message("assistant"):
            st.markdown(response.text)
            if response.usage_metadata:
                u = response.usage_metadata
                cached_val = getattr(u, "cached_content_token_count", 0)
                cached_info = f" [Cached: {cached_val}]" if cached_val else ""
                st.caption(f"Tokens: {u.total_token_count} (Prompt: {u.prompt_token_count}{cached_info}, Response: {u.candidates_token_count})")
        
        # F. Add to local history
        msg_data = {"role": "assistant", "content": response.text}
        if response.usage_metadata:
            u = response.usage_metadata
            msg_data["usage"] = {
                "prompt_tokens": u.prompt_token_count,
                "candidates_tokens": u.candidates_token_count,
                "total_tokens": u.total_token_count,
                "cached_tokens": getattr(u, "cached_content_token_count", 0)
            }
        st.session_state.messages.append(msg_data)

    except Exception as e:
        st.error(f"An error occurred: {e}")
