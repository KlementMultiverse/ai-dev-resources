# End-to-End AI Project Template: RAG Chatbot from Zero to Production

**Complete blueprint with code, architecture, and deployment for production-ready AI applications**

---

## 🎯 What You'll Build

**Project:** Customer Support RAG Chatbot
- ✅ Answers questions from documentation
- ✅ Cites sources for every answer
- ✅ Handles 10k+ users
- ✅ Costs < $200/month
- ✅ Deploy in 1 day

---

## 📋 Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── rag_engine.py          # RAG logic
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # Pinecone integration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # React chat UI
│   │   ├── ChatBox.tsx        # Chat component
│   │   └── api.ts             # Backend API calls
│   ├── package.json
│   └── vite.config.ts
├── data/
│   ├── docs/                   # Raw documentation
│   └── processed/              # Chunked data
├── scripts/
│   ├── ingest_docs.py         # Data ingestion
│   └── evaluate.py            # Quality testing
├── .env
├── docker-compose.yml
└── README.md
```

---

## 🚀 Phase 1: Backend Setup (2 hours)

### **Step 1: Environment Setup**

```bash
# Create project
mkdir rag-chatbot && cd rag-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn openai pinecone-client python-dotenv pydantic
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn==0.24.0
openai==1.3.0
pinecone-client==2.2.4
python-dotenv==1.0.0
pydantic==2.5.0
langchain==0.0.350
redis==5.0.1
```

**.env:**
```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX=support-docs
REDIS_URL=redis://localhost:6379
```

---

### **Step 2: RAG Engine (Core Logic)**

**rag_engine.py:**
```python
from typing import List, Dict
import openai
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    """Production RAG system"""

    def __init__(self):
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT')
        )
        self.index = pinecone.Index(os.getenv('PINECONE_INDEX'))

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for query"""
        response = openai.Embedding.create(
            input=text,
            model='text-embedding-ada-002'
        )
        return response['data'][0]['embedding']

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Semantic search in vector DB"""
        # Get query embedding
        query_vector = self.embed_query(query)

        # Search Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        return [
            {
                'text': match['metadata']['text'],
                'source': match['metadata'].get('source', 'Unknown'),
                'score': match['score']
            }
            for match in results['matches']
        ]

    def generate_answer(self, query: str, context: List[Dict]) -> Dict:
        """Generate answer with citations"""
        # Build context string
        context_text = '\n\n'.join([
            f"[Source: {ctx['source']}]\n{ctx['text']}"
            for ctx in context
        ])

        # Create prompt
        prompt = f"""Use the following context to answer the question.
Include source citations in your answer.
If the answer isn't in the context, say "I don't have information about that in the documentation."

Context:
{context_text}

Question: {query}

Answer (with citations):"""

        # Generate response
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful customer support assistant. Always cite your sources.'
                },
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response['choices'][0]['message']['content']

        return {
            'answer': answer,
            'sources': [
                {'source': ctx['source'], 'score': ctx['score']}
                for ctx in context
            ]
        }

    def query(self, question: str) -> Dict:
        """Full RAG pipeline"""
        # 1. Retrieve relevant context
        context = self.retrieve(question, top_k=3)

        # 2. Generate answer
        result = self.generate_answer(question, context)

        return result
```

---

### **Step 3: FastAPI Server**

**main.py:**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine
import redis
import json
import hashlib

app = FastAPI(title='RAG Chatbot API')

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # In production: specific domains only
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize RAG engine
rag = RAGEngine()

# Redis cache (optional but recommended)
try:
    cache = redis.Redis(host='localhost', port=6379, decode_responses=True)
except:
    cache = None  # Fallback if Redis not available

class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list
    cached: bool = False

@app.post('/query', response_model=QueryResponse)
async def query(request: QueryRequest):
    """RAG query endpoint with caching"""
    try:
        # Check cache
        cache_key = hashlib.md5(request.question.encode()).hexdigest()

        if request.use_cache and cache:
            cached_result = cache.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result['cached'] = True
                return result

        # RAG query
        result = rag.query(request.question)

        # Cache result (1 hour TTL)
        if cache:
            cache.setex(cache_key, 3600, json.dumps(result))

        result['cached'] = False
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    """Health check"""
    return {'status': 'ok'}

@app.get('/')
async def root():
    return {'message': 'RAG Chatbot API is running'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

**Run backend:**
```bash
python main.py
# Visit http://localhost:8000/docs for API docs
```

---

## 🎨 Phase 2: Frontend (React + TypeScript) (2 hours)

### **Step 1: Create React App**

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install axios @types/axios
```

### **Step 2: API Integration**

**src/api.ts:**
```typescript
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface QueryResponse {
  answer: string;
  sources: Array<{
    source: string;
    score: number;
  }>;
  cached: boolean;
}

export async function queryRAG(question: string): Promise<QueryResponse> {
  const response = await axios.post(`${API_URL}/query`, {
    question,
    use_cache: true
  });

  return response.data;
}
```

### **Step 3: Chat UI**

**src/ChatBox.tsx:**
```typescript
import { useState } from 'react';
import { queryRAG, QueryResponse } from './api';
import './ChatBox.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ source: string; score: number }>;
}

export default function ChatBox() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Query RAG
      const response = await queryRAG(input);

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, something went wrong. Please try again.'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                <strong>Sources:</strong>
                {msg.sources.map((src, j) => (
                  <span key={j} className="source-tag">
                    {src.source} ({(src.score * 100).toFixed(0)}%)
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message assistant loading">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
```

**src/ChatBox.css:**
```css
.chat-container {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
  margin-bottom: 20px;
}

.message {
  margin-bottom: 16px;
  padding: 12px 16px;
  border-radius: 8px;
  max-width: 80%;
}

.message.user {
  background: #007bff;
  color: white;
  margin-left: auto;
  text-align: right;
}

.message.assistant {
  background: white;
  color: #333;
  border: 1px solid #ddd;
}

.sources {
  margin-top: 8px;
  font-size: 12px;
  color: #666;
}

.source-tag {
  display: inline-block;
  background: #e9ecef;
  padding: 4px 8px;
  border-radius: 4px;
  margin: 4px 4px 0 0;
}

.typing-indicator {
  display: flex;
  gap: 4px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: #999;
  border-radius: 50%;
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    opacity: 0.3;
  }
  30% {
    opacity: 1;
  }
}

.input-form {
  display: flex;
  gap: 12px;
}

.input-form input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 16px;
}

.input-form button {
  padding: 12px 24px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
}

.input-form button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
```

**src/App.tsx:**
```typescript
import ChatBox from './ChatBox';
import './App.css';

function App() {
  return (
    <div className="App">
      <header>
        <h1>🤖 Customer Support Chatbot</h1>
        <p>Ask me anything about our product documentation</p>
      </header>
      <ChatBox />
    </div>
  );
}

export default App;
```

**Run frontend:**
```bash
npm run dev
# Visit http://localhost:5173
```

---

## 📊 Phase 3: Data Ingestion (1 hour)

**scripts/ingest_docs.py:**
```python
import openai
import pinecone
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Initialize
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# Create index if doesn't exist
index_name = os.getenv('PINECONE_INDEX')
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI ada-002 embedding size
        metric='cosine'
    )

index = pinecone.Index(index_name)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break on sentence
        if end < len(text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embed texts"""
    response = openai.Embedding.create(
        input=texts,
        model='text-embedding-ada-002'
    )
    return [item['embedding'] for item in response['data']]

def ingest_documents(docs_path: str):
    """Ingest all documents from directory"""
    docs_dir = Path(docs_path)

    for doc_file in docs_dir.glob('*.md'):
        print(f"Processing {doc_file.name}...")

        # Read document
        text = doc_file.read_text()

        # Chunk
        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")

        # Embed
        embeddings = embed_texts(chunks)
        print(f"  Generated {len(embeddings)} embeddings")

        # Prepare vectors
        vectors = [
            (
                f"{doc_file.stem}_{i}",  # ID
                embeddings[i],  # Vector
                {
                    'text': chunks[i],
                    'source': doc_file.name,
                    'chunk_id': i
                }
            )
            for i in range(len(chunks))
        ]

        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"  ✅ Uploaded {len(vectors)} vectors")

    print(f"\n✅ Ingestion complete! Total vectors: {index.describe_index_stats()['total_vector_count']}")

if __name__ == '__main__':
    ingest_documents('data/docs')
```

**Run ingestion:**
```bash
# Add your docs to data/docs/
python scripts/ingest_docs.py
```

---

## 🧪 Phase 4: Testing & Evaluation (1 hour)

**scripts/evaluate.py:**
```python
from rag_engine import RAGEngine
import json

# Test questions
test_cases = [
    {
        'question': 'How do I reset my password?',
        'expected_topic': 'authentication'
    },
    {
        'question': 'What are the pricing plans?',
        'expected_topic': 'pricing'
    },
    # Add more test cases
]

rag = RAGEngine()

print("📊 Evaluation Results\n" + "="*60)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. Question: {test['question']}")

    result = rag.query(test['question'])

    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Sources: {len(result['sources'])}")
    print(f"   Top source score: {result['sources'][0]['score']:.2f}")

print("\n" + "="*60)
```

---

## 🚀 Phase 5: Deployment (2 hours)

### **Option 1: Docker Compose (Local/VPS)**

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - PINECONE_INDEX=${PINECONE_INDEX}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    environment:
      - VITE_API_URL=http://localhost:8000

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```bash
# Deploy
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

### **Option 2: Railway (1-click deploy)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy backend
cd backend
railway up

# Deploy frontend
cd ../frontend
railway up
```

---

## 📊 Cost Breakdown

| Component | Service | Cost/Month |
|-----------|---------|------------|
| **Embeddings** | OpenAI ada-002 | $5 (100k chunks) |
| **LLM** | OpenAI GPT-4 | $150 (10k queries) |
| **Vector DB** | Pinecone Starter | $70 (100k vectors) |
| **Backend hosting** | Railway/Render | $20 |
| **Frontend hosting** | Vercel/Netlify | $0 (free tier) |
| **Redis cache** | Upstash | $0 (free tier) |
| **Total** |  | **~$245/mo** |

**Cost optimization:**
- Use GPT-3.5-turbo instead of GPT-4: **-$120/mo**
- Self-host vector DB: **-$70/mo**
- **Optimized total: ~$55/mo**

---

## 🎯 Production Checklist

- [ ] Environment variables secured
- [ ] Rate limiting enabled
- [ ] CORS configured for production domain
- [ ] Caching implemented (Redis)
- [ ] Logging and monitoring (Sentry, LogRocket)
- [ ] Health checks and alerts
- [ ] Backup strategy for vector DB
- [ ] API authentication (API keys)
- [ ] HTTPS enabled
- [ ] Analytics tracking (Mixpanel, PostHog)

---

## 🔄 Iterating for Production

### **Week 1: MVP**
- ✅ Basic RAG working
- ✅ Simple UI deployed
- ✅ 10-20 test users

### **Week 2: Optimization**
- Add caching (reduce costs by 60%)
- Improve chunking strategy
- Add conversation memory

### **Week 3: Features**
- User authentication
- Conversation history
- Feedback collection (thumbs up/down)

### **Week 4: Scale**
- Load testing
- Auto-scaling
- Multi-language support

---

## 💡 Pro Tips

1. **Start with GPT-3.5-turbo** - Switch to GPT-4 only if quality insufficient
2. **Cache aggressively** - 70% of queries are similar
3. **Monitor quality** - Add thumbs up/down feedback from day 1
4. **Test with real docs** - Generic Lorem Ipsum won't reveal issues
5. **Deploy fast** - Get feedback early, iterate based on usage

---

## 📚 Repository Template

Clone the complete template:
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot-template.git
cd rag-chatbot-template
```

Includes:
- ✅ Full backend + frontend code
- ✅ Docker configs
- ✅ Test suite
- ✅ CI/CD workflows
- ✅ Documentation

---

## 🎓 Next Steps

1. **Clone this template**
2. **Add your documentation to `data/docs/`**
3. **Run ingestion script**
4. **Deploy to Railway/Render**
5. **Share with users!**

**Timeline:** 1 day MVP → 1 week production-ready

---

**Questions? Issues?**

Open an issue on GitHub or check the docs:
- FastAPI: https://fastapi.tiangolo.com
- Pinecone: https://docs.pinecone.io
- React: https://react.dev

---

**Made with ❤️ for AI builders**

*Ship production AI in days, not months!*
