# 🎬 YouTube Chatbot and Recommender

An intelligent chatbot that allows you to **ask questions about any YouTube video** and get answers directly from its transcript using **Retrieval-Augmented Generation (RAG)**.


##  Features

*  Load any YouTube video
*  Ask questions about the video
*  Context-aware answers using transcript
*  Smart retrieval using FAISS vector search
*  Timestamp-based video references
*  Auto video recommendations if answer not found
*  Multi-chat support


##  Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM:** Groq (LLaMA 3.3)
* **Embeddings:** HuggingFace (MiniLM)
* **Vector DB:** FAISS
* **Transcript API:** youtube-transcript-api
* **Search:** YouTube Search Python


##  How It Works

1. User enters a YouTube URL
2. Transcript is fetched
3. Text is split into chunks
4. Chunks are converted into embeddings
5. Stored in FAISS vector database
6. User query is matched with relevant chunks
7. LLM generates answer using retrieved context
8. If answer not found → recommends related videos


##  Project Structure

```
├── app.py          
├── backend.py      
├── .env            
└── README.md
```


##  Setup Instructions

### 1. Clone the repository

```
[git clone https://github.com/your-username/youtube-rag-chatbot.git
cd youtube-rag-chatbot](https://github.com/Nitya-Pahwa/Youtube-ChatBot-and-Recommender.git)
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```


### 4. Run the App

```
streamlit run app.py
```


##  Usage

1. Paste a YouTube video URL
2. Click **Load Video**
3. Ask questions about the video
4. View answers with timestamps
5. Get recommendations if needed


## Glimpses of Project  
<img width="1920" height="894" alt="Screenshot (1586)" src="https://github.com/user-attachments/assets/aeccaa3d-75aa-4145-83c6-488697462057" /><br>
<img width="1920" height="904" alt="Screenshot (1587)" src="https://github.com/user-attachments/assets/fffcf41f-a0e3-49fc-80d4-1582331b9c3c" /><br>
<img width="1920" height="890" alt="Screenshot (1588)" src="https://github.com/user-attachments/assets/a7cb3a8b-7c62-4e34-8d66-5ccd59524400" /><br>
<img width="1920" height="900" alt="Screenshot (1589)" src="https://github.com/user-attachments/assets/0e2d1160-6c43-41e8-af59-575c646d5dbc" /><br>
<img width="1920" height="910" alt="Screenshot (1590)" src="https://github.com/user-attachments/assets/8e61712b-7acc-4dc9-a12e-01c5f430f73e" /><br>
