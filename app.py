import streamlit as st
from config import DOCUMENTS_DIR
from loader import get_document_chunks
from vectorstore import get_vectorstore, get_retriever
from chain import create_rag_chain


# =====================================================================
# 1. Page Configuration
# =====================================================================
st.set_page_config(
    page_title="Love2D AI Assistant",
    page_icon="🎮",
    layout="centered",
)

st.title("🎮 Love2D Game Dev Assistant")
st.write("Ask any questions about Love2D game development based on your documentation.")


# =====================================================================
# 2. Cache the RAG pipeline
# ---------------------------------------------------------------------
# Why @st.cache_resource?
# In Streamlit, the entire script re-runs every time a user types a message.
# @st.cache_resource makes sure we only load the documents, vector store,
# and RAG chain ONCE into memory, instead of reloading on every message.
# =====================================================================
@st.cache_resource(show_spinner="Loading documentation & vector store...")
def load_rag_pipeline():
    try:
        # Try loading existing cached FAISS index from disk
        vs = get_vectorstore()
    except ValueError:
        # If cache doesn't exist yet, load PDFs and build index
        chunks = get_document_chunks(DOCUMENTS_DIR)
        vs = get_vectorstore(chunks)

    retriever = get_retriever(vs)
    rag_chain = create_rag_chain(retriever)
    return rag_chain


# Initialize the chain
rag_chain = load_rag_pipeline()


# =====================================================================
# 3. Chat History (Session State)
# ---------------------------------------------------------------------
# Streamlit clears local variables on each re-run.
# st.session_state allows us to keep the conversation history in memory.
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Love2D AI assistant. Ask me anything about physics, animations, tables, or game loops!",
        }
    ]

# Display previous conversation messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================================
# 4. Handle New User Input
# =====================================================================
# st.chat_input shows a fixed input bar at the bottom of the screen
if prompt := st.chat_input("Ask a question about Love2D..."):
    # 1. Add user message to session history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate answer with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching docs and generating answer..."):
            try:
                response = rag_chain.invoke(prompt)
                if response and response.answers:
                    formatted_answers = []
                    for ans in response.answers:
                        ans_text = ans.get("text", "").strip()
                        ans_context = ans.get("context", "").strip()

                        # Safety guard: Check if the response indicates out-of-context or unanswerable query
                        negative_cues = (
                            "does not contain",
                            "not contain information",
                            "does not provide",
                            "not mentioned",
                            "not enough information",
                            "cannot be answered",
                            "not found in the documentation",
                            "no information",
                            "out of scope",
                        )
                        is_out_of_context = any(cue in ans_text.lower() for cue in negative_cues)

                        part = ans_text
                        # Only show reference if context exists, is not empty, and query wasn't out-of-scope
                        if ans_context and not is_out_of_context and ans_context not in ans_text:
                            if any(k in ans_context for k in ("function", "local", "end", "return", "\n")):
                                part += f"\n\n**Supporting Code / Reference:**\n```lua\n{ans_context}\n```"
                            else:
                                part += f"\n\n> **Reference:** {ans_context}"
                        formatted_answers.append(part)

                    answer_text = "\n\n---\n\n".join(formatted_answers)
                else:
                    answer_text = "I couldn't find an answer in the documentation."
            except Exception as e:
                answer_text = f"An error occurred: {e}"

            st.markdown(answer_text)

    # 3. Save assistant response into session history
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
