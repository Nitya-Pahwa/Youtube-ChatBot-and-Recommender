# Streamlit-based frontend for YouTube RAG chatbot
# Handles UI, chat management, and interaction with backend RAG pipeline

import streamlit as st
import os
from dotenv import load_dotenv
from backend import build_rag_chain

load_dotenv()

st.set_page_config(
    page_title="YouTube Chatbot and Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 YouTube ChatBot and Recommender")

# ================= SESSION INIT =================
# This block initializes chat sessions and maintains multiple chat states

# Initialize storage for multiple chats
if "chats" not in st.session_state:
    st.session_state.chats = {}

# Create default chat if none exists
if "current_chat" not in st.session_state:
    chat_id = "chat_1"
    st.session_state.chats[chat_id] = {
        "title": "Untitled",
        "messages": [],
        "video_loaded": False,
        "chain": None,
        "video_url": ""
    }
    st.session_state.current_chat = chat_id

current_chat_id = st.session_state.current_chat
current_chat = st.session_state.chats[current_chat_id]

# ================= SIDEBAR =================
# Sidebar handles video loading and chat switching

with st.sidebar:

    # ============ LOAD VIDEO (TOP) ============
    st.header("📺 Load Video")

    # Input field for YouTube video URL
    video_url = st.text_input(
        "YouTube URL",
        value=current_chat.get("video_url", ""),
        key=f"video_input_{current_chat_id}"
    )

    # When user clicks "Load Video", build RAG pipeline using transcript
    if st.button(" Load Video", use_container_width=True):

        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please set GROQ_API_KEY in .env file")

        elif not video_url:
            st.warning("Please enter a YouTube URL")

        else:
            with st.spinner("Loading transcript..."):
                try:
                    # Build RAG chain using backend (transcript + embeddings + LLM)
                    chain = build_rag_chain(video_url)

                    current_chat["chain"] = chain
                    current_chat["video_loaded"] = True
                    current_chat["messages"] = []
                    current_chat["video_url"] = video_url
                    current_chat["title"] = "Untitled"

                    st.success("Video loaded successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # ============ CHATS ============
    # Allows creating and switching between multiple chats
    st.header("💬 Chats")

    # New Chat (ADD AT TOP)
    # Create a new chat session and make it active
    if st.button("➕ New Chat", use_container_width=True):

        new_chat_number = len(st.session_state.chats) + 1
        new_chat_id = f"chat_{new_chat_number}"

        st.session_state.chats = {
            new_chat_id: {
                "title": "Untitled",
                "messages": [],
                "video_loaded": False,
                "chain": None,
                "video_url": ""
            },
            **st.session_state.chats
        }

        st.session_state.current_chat = new_chat_id
        st.rerun()

    st.markdown("---")

    # Display all previous chats in sidebar
    for chat_id, chat_data in st.session_state.chats.items():
        if st.button(chat_data["title"], use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()

# ================= MAIN CHAT =================
# Handles conversation display and user interaction

current_chat = st.session_state.chats[st.session_state.current_chat]

if current_chat["video_loaded"]:

    st.header("💬 Chat")

    # ===== DISPLAY CHAT HISTORY =====
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            # Show sources
            if msg.get("sources"):
                st.markdown("### Relevant part of the video:")
                for link in msg["sources"]:
                    st.markdown(f"- ▶️ [Jump to Timestamp]({link})")

            # Show recommended videos if query is not in transcript
            if msg.get("recommendations"):
                st.warning("This question is not covered in this video.")
                st.markdown("### 🔎 Recommended Videos")

                for video in msg["recommendations"]:
                    st.markdown(f"**[{video['title']}]({video['link']})**")
                    st.caption(
                        f"Channel: {video['channel']} | Duration: {video['duration']}"
                    )
                    st.image(video["thumbnail"], width=250)
                    st.markdown("---")

    # ===== USER INPUT =====
    user_input = st.chat_input("Ask something about the video...")

    if user_input:

        # FIX: UPDATE TITLE ON FIRST MESSAGE ONLY
        if len(current_chat["messages"]) == 0:
            current_chat["title"] = user_input[:40] + (
                "..." if len(user_input) > 40 else ""
            )

        # Store user message
        current_chat["messages"].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare history text
        chat_history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in current_chat["messages"]]
        )

        result = current_chat["chain"](user_input, chat_history_text)

        with st.chat_message("assistant"):

            if result["type"] == "answer":

                st.markdown(result["answer"])

                if result["sources"]:
                    st.markdown("### 📍 Relevant part of the video:")
                    for link in result["sources"]:
                        st.markdown(f"- ▶️ [Jump to Timestamp]({link})")

                current_chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    }
                )

            elif result["type"] == "recommendation":

                st.warning("This question is not covered in this video.")
                st.markdown("### 🔎 Recommended Videos")

                for video in result["videos"]:
                    st.markdown(f"**[{video['title']}]({video['link']})**")
                    st.caption(
                        f"Channel: {video['channel']} | Duration: {video['duration']}"
                    )
                    st.image(video["thumbnail"], width=250)
                    st.markdown("---")

                current_chat["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Recommended videos shown above.",
                        "recommendations": result["videos"]
                    }
                )

else:
    st.info(" Please load a YouTube video from the sidebar to start chatting.")