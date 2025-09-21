# Adaptive Retrieval-Augmented Generation Learning (RAGL) System  
🎥 **[Project Demo on YouTube](https://youtu.be/gpIohWJ3XwA)**

---

## 📘 Project Overview  
The **Adaptive RAGL System** is an intelligent learning assistant built on the Retrieval-Augmented Generation (RAG) framework. It continuously learns from user interactions, updates its knowledge base in real time, and delivers contextual responses using advanced Large Language Models (LLMs).  


---

## 🔑 Key Features

### 🗃️ Multi-Format Document Processing  
Supports ingestion and processing of diverse document formats:
- PDF  
- Word (DOC, DOCX)  
- Excel (XLS, XLSX)  
- PowerPoint (PPT, PPTX)  
- CSV, JSON, XML, HTML  
- Plain text (TXT, MD)

### 🧠 Intelligent Learning Capabilities  
- Continuous, incremental learning from documents and user inputs  
- Adaptive retrieval using vector search  
- Feedback-based system improvement  
- Metadata tracking and event logging

### 💻 User-Friendly Interface  
- Streamlit-based real-time chat UI  
- Upload and manage documents via sidebar  
- Learning events and session history tracking  

---

## ⚙️ Technical Architecture  

### Core Components  
**Vector Database (FAISS)**  
- Efficient similarity search using vector embeddings  
- Dynamic index updates  

**Document Processor**  
- Format-specific loaders and parsers  
- Text chunking and metadata extraction  

**RAG System**  
- Combines retrieval and generation for accurate answers  
- Updates knowledge dynamically  

**LLM Integration**  
- Google Gemini AI for generation  
- LangChain for orchestration and prompt tuning  

---

## 🧰 Technology Stack

| Layer     | Tools Used                                                 |
|-----------|------------------------------------------------------------|
| Backend   | Python 3.12, FAISS, LangChain, PyPDF2, python-docx, pandas, python-pptx |
| Frontend  | Streamlit, Custom CSS                                      |
| LLM       | Google Gemini (via API)                                    |

---

## 🚀 Setup Instructions

### 📋 Prerequisites  
- Python 3.12+  
- Google Gemini API Key  
- Git

### 📦 Installation  
```bash
git clone [your-repository-url]
cd [repository-name]
pip install -r requirements.txt
```

# 🔐 Environment Configuration
Create a `.env` file in the root directory and add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

# ▶️ Running the Application

```bash
python app.py
```
Then visit: http://localhost:8501

## 📚 Usage Guide

### 📤 Upload Documents  
Upload files via the sidebar. The system extracts content and indexes it for retrieval.

### 💬 Chat and Retrieve  
Ask questions using the chat interface. The system retrieves relevant chunks and generates answers using the Gemini LLM.

### 📈 Feedback and Learning  
Provide relevance feedback on the retrieved documents. The system uses it to improve future results and logs learning events.

### 🧾 Session Management  
Save and load past conversations, track new knowledge, and manage your knowledge base.


## 📁 Project Structure

```bash
ml/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
├── README.md              # Documentation
├── data/
│   ├── sessions/          # Saved sessions
│   └── vector_db/         # FAISS index and vectors
└── src/
    ├── components/
    │   ├── ragl_system.py
    │   ├── vector_db.py
    │   └── learner.py
    └── utils/
        └── config.py

```
## 🔮 Future Enhancements

- 🧾 OCR and image-based text extraction  
- 🧠 Personalized learning profiles  
- ⚡ Faster response times with caching  
- 🗃️ Better document classification  
- 📡 Real-time API support  

## 🤝 Contributing

Open to feature suggestions, bug reports, or pull requests. Let’s build together!

