# Backend logic for YouTube chatbot and Recommender
# Handles transcript extraction, embedding, vector search, and LLM response generation
import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, CouldNotRetrieveTranscript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from youtubesearchpython import VideosSearch

load_dotenv()

# Extract YouTube video ID from different URL formats
def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract YouTube video ID from URL.")

# Search YouTube for related videos if answer is not found in transcript
def search_youtube_videos(query):
    videos_search = VideosSearch(query, limit=5)
    results = videos_search.result()

    videos = []
    for item in results["result"]:
        videos.append({
            "title": item["title"],
            "link": item["link"],
            "channel": item["channel"]["name"],
            "duration": item.get("duration", "N/A"),
            "thumbnail": item["thumbnails"][0]["url"]
        })

    return videos

# Main function to build RAG pipeline for a given YouTube video
def build_rag_chain(youtube_url: str):

    video_id = extract_video_id(youtube_url)

    # Fetch transcript using YouTubeTranscriptApi (v1.x method)
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)

        transcript_data = []
        for snippet in fetched:
            # Store transcript text along with timestamp
            transcript_data.append({
                "text": snippet.text,
                "start": snippet.start
            })

    except CouldNotRetrieveTranscript:
        raise ValueError("This video does not have English captions.")
    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {str(e)}")

    # Combine all transcript text into one large string
    full_text = " ".join([chunk["text"] for chunk in transcript_data])

    # Split transcript into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([full_text])

    # Attach video ID and timestamp metadata to each chunk
    for i, doc in enumerate(docs):
        doc.metadata["video_id"] = video_id
        doc.metadata["start"] = transcript_data[min(i, len(transcript_data)-1)]["start"]

    # Convert text chunks into vector embeddings and store in FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Initialize Groq LLM (LLaMA model)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    # Prompt template to control LLM behavior
    prompt = PromptTemplate(
        template="""
You are a helpful assistant answering questions about a YouTube video.

Use ONLY the provided transcript context.

If the answer is not present in the context, say:
"I don't know based on the provided video transcript."

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["chat_history", "context", "question"]
    )

    # Function to process user query using RAG pipeline
    def ask_question(question, chat_history):

        # Retrieve top relevant chunks based on similarity
        docs_with_scores = vector_store.similarity_search_with_score(question, k=4)

        context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])

        formatted_prompt = prompt.format(
            chat_history=chat_history,
            context=context,
            question=question
        )

        response = llm.invoke(formatted_prompt)
        answer = response.content.strip()

        # If not found go to  recommend
        if answer.lower().startswith("i don't know"):
            recommendations = search_youtube_videos(question)
            return {
                "type": "recommendation",
                "videos": recommendations
            }

        # Otherwise answer normally
        sources = []
        for doc, _ in docs_with_scores:
            start_time = int(doc.metadata.get("start", 0))
            link = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
            sources.append(link)

        return {
            "type": "answer",
            "answer": answer,
            "sources": list(set(sources))
        }

    return ask_question