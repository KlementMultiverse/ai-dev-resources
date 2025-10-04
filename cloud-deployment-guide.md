# Cloud Deployment Guide: AI Models on AWS, GCP & Azure

**Production-ready deployment strategies for LLMs, RAG systems, and ML models**

---

## 🎯 Quick Platform Comparison

| Platform | Best For | Cost | Ease of Use | AI-Specific Features |
|----------|----------|------|-------------|---------------------|
| **AWS** | Enterprise, Flexibility | 💰💰💰 | ⭐⭐⭐ | SageMaker, Bedrock |
| **GCP** | ML/AI Innovation | 💰💰 | ⭐⭐⭐⭐ | Vertex AI, TPUs |
| **Azure** | Microsoft Stack | 💰💰💰 | ⭐⭐⭐ | OpenAI Service |
| **Hugging Face** | Prototyping | 💰 | ⭐⭐⭐⭐⭐ | Inference API |
| **Railway/Render** | Small Projects | 💰 | ⭐⭐⭐⭐⭐ | Simple deploy |

---

## 🚀 Deployment Architecture Patterns

### **Pattern 1: API-Only (No Model Hosting)**

**When:** Using OpenAI/Anthropic APIs for LLM
**Cost:** ~$100-500/mo (API calls only)
**Complexity:** ⭐ (Easiest)

```
┌─────────┐      ┌──────────────┐      ┌─────────────┐
│ Frontend│─────▶│  Your API    │─────▶│ OpenAI API  │
│ (React) │      │ (FastAPI)    │      │ (GPT-4)     │
└─────────┘      └──────────────┘      └─────────────┘
                        │
                        ▼
                  ┌──────────────┐
                  │ Vector DB    │
                  │ (Pinecone)   │
                  └──────────────┘
```

**Best for:** RAG chatbots, MVP products, small scale

---

### **Pattern 2: Self-Hosted Models**

**When:** Need cost control at scale, data privacy
**Cost:** ~$500-2000/mo (compute + storage)
**Complexity:** ⭐⭐⭐⭐

```
┌─────────┐      ┌──────────────┐      ┌─────────────┐
│ Frontend│─────▶│  Load        │─────▶│ GPU Server  │
│         │      │  Balancer    │      │ (Llama 2)   │
└─────────┘      └──────────────┘      └─────────────┘
                        │                      │
                        │                      ▼
                        │                ┌──────────────┐
                        │                │ GPU Server   │
                        │                │ (Llama 2)    │
                        │                └──────────────┘
                        ▼
                  ┌──────────────┐
                  │ Vector DB    │
                  │ (Self-hosted)│
                  └──────────────┘
```

**Best for:** High volume, cost-sensitive, compliance requirements

---

## ☁️ AWS Deployment (Complete Guide)

### **Option 1: Lambda + API Gateway (Serverless RAG)**

**Cost:** ~$50-200/mo
**Pros:** Auto-scaling, pay-per-use
**Cons:** 15min timeout, cold starts

```bash
# Project structure
my-rag-app/
├── lambda_function.py
├── requirements.txt
└── .env
```

**lambda_function.py:**
```python
import json
import os
import openai
import pinecone

# Initialize once (outside handler for warm starts)
openai.api_key = os.environ['OPENAI_API_KEY']
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment='us-west1-gcp')
index = pinecone.Index('my-index')

def lambda_handler(event, context):
    """RAG endpoint"""
    try:
        # Parse request
        body = json.loads(event['body'])
        query = body['query']

        # Embed query
        query_embedding = openai.Embedding.create(
            input=query,
            model='text-embedding-ada-002'
        )['data'][0]['embedding']

        # Retrieve from Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        # Build context
        context = '\n\n'.join([
            match['metadata']['text']
            for match in results['matches']
        ])

        # Generate answer
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': 'Answer based on context.'},
                {'role': 'user', 'content': f'Context:\n{context}\n\nQuestion: {query}'}
            ],
            temperature=0
        )

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'answer': response['choices'][0]['message']['content'],
                'sources': len(results['matches'])
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**Deploy:**
```bash
# 1. Install dependencies to package/
pip install openai pinecone-client -t package/

# 2. Create deployment package
cd package
zip -r ../lambda_deployment.zip .
cd ..
zip -g lambda_deployment.zip lambda_function.py

# 3. Deploy via AWS CLI
aws lambda create-function \
  --function-name rag-chatbot \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_deployment.zip \
  --timeout 60 \
  --memory-size 512 \
  --environment Variables="{OPENAI_API_KEY=sk-...,PINECONE_API_KEY=...}"

# 4. Create API Gateway endpoint
aws apigatewayv2 create-api \
  --name rag-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:YOUR_ACCOUNT:function:rag-chatbot
```

**Test:**
```bash
curl -X POST https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

---

### **Option 2: EC2 + Docker (Self-Hosted Model)**

**Cost:** ~$500-1500/mo (g4dn.xlarge GPU instance)
**Pros:** Full control, any model
**Cons:** Manual scaling, management overhead

```bash
# 1. Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-groups rag-server-sg

# 2. SSH into instance
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 3. Install Docker & NVIDIA runtime
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install -y nvidia-docker2
```

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose API port
EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**main.py (FastAPI app):**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model once at startup
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

class QueryRequest(BaseModel):
    query: str
    max_length: int = 512

@app.post('/generate')
def generate(request: QueryRequest):
    """Generate text with Llama 2"""
    inputs = tokenizer(request.query, return_tensors='pt').to('cuda')

    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {'response': response}

@app.get('/health')
def health():
    return {'status': 'ok'}
```

**Deploy:**
```bash
# Build and run
docker build -t rag-server .
docker run -d --gpus all -p 8000:8000 rag-server

# Test
curl http://YOUR_EC2_IP:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

---

## 🌐 GCP Deployment (Vertex AI)

### **Vertex AI + Cloud Run (Managed)**

**Cost:** ~$100-400/mo
**Pros:** Fully managed, auto-scaling
**Cons:** GCP-specific

```bash
# 1. Enable APIs
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com

# 2. Create service account
gcloud iam service-accounts create rag-service

# 3. Deploy to Cloud Run
gcloud run deploy rag-api \
  --image gcr.io/YOUR_PROJECT/rag-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

**Using Vertex AI for embeddings:**
```python
from google.cloud import aiplatform

aiplatform.init(project='your-project', location='us-central1')

def embed_text(texts: list) -> list:
    """Use Vertex AI text embeddings"""
    endpoint = aiplatform.Endpoint('text-embedding-endpoint')

    predictions = endpoint.predict(instances=[{'content': t} for t in texts])

    return [pred['embeddings'] for pred in predictions.predictions]
```

---

## 🔷 Azure Deployment (Azure OpenAI Service)

### **Why Azure?**
- ✅ Access to GPT-4 without OpenAI waitlist
- ✅ Enterprise compliance (HIPAA, SOC 2)
- ✅ VNET integration for security
- ❌ More expensive than OpenAI direct

```python
import openai

# Configure Azure endpoint
openai.api_type = "azure"
openai.api_base = "https://YOUR_RESOURCE.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = "YOUR_AZURE_KEY"

# Use GPT-4 (same code as OpenAI!)
response = openai.ChatCompletion.create(
    engine="gpt-4",  # Your deployment name
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Deploy FastAPI to Azure App Service:**
```bash
# 1. Create App Service
az webapp up \
  --name rag-chatbot \
  --runtime "PYTHON:3.11" \
  --sku B1

# 2. Set environment variables
az webapp config appsettings set \
  --name rag-chatbot \
  --settings \
    OPENAI_API_KEY=your-azure-key \
    OPENAI_API_BASE=https://YOUR_RESOURCE.openai.azure.com/
```

---

## 🎯 Production Checklist

### **Security:**
- [ ] API keys in environment variables (never hardcode!)
- [ ] HTTPS only (SSL/TLS certificates)
- [ ] Rate limiting (prevent abuse)
- [ ] Authentication (API keys or OAuth)
- [ ] Input validation (prevent injection attacks)

### **Performance:**
- [ ] Caching (Redis for frequent queries)
- [ ] Connection pooling (reuse DB connections)
- [ ] Async operations (FastAPI async endpoints)
- [ ] CDN for static assets
- [ ] Health checks (monitor uptime)

### **Cost Optimization:**
- [ ] Auto-scaling rules (scale down when idle)
- [ ] Spot instances for batch jobs (70% cheaper)
- [ ] Caching to reduce API calls
- [ ] Monitor usage (set billing alerts)
- [ ] Choose right instance size (don't overprovision)

### **Monitoring:**
- [ ] Logging (CloudWatch, Stackdriver, Application Insights)
- [ ] Metrics (latency, error rate, throughput)
- [ ] Alerts (notify on errors/high latency)
- [ ] Distributed tracing (OpenTelemetry)

---

## 💰 Cost Breakdown (1000 users/day)

### **Scenario: RAG Chatbot (10 queries/user/day)**

| Component | AWS | GCP | Azure |
|-----------|-----|-----|-------|
| **Compute** | Lambda: $50 | Cloud Run: $40 | App Service: $70 |
| **Vector DB** | Pinecone: $70 | Pinecone: $70 | Pinecone: $70 |
| **LLM API** | OpenAI: $300 | OpenAI: $300 | Azure OpenAI: $400 |
| **Storage** | S3: $10 | GCS: $10 | Blob: $15 |
| **Total/mo** | **$430** | **$420** | **$555** |

**Winner:** GCP (slightly cheaper compute)

---

## 🚀 Quick Start Templates

### **1. Railway (Easiest - 1 click deploy)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy (detects Python automatically!)
railway up

# Get URL
railway domain
```

**Cost:** $5-20/mo for small apps

---

### **2. Render (Simple, affordable)**

```yaml
# render.yaml
services:
  - type: web
    name: rag-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
```

```bash
# Connect GitHub repo, Render auto-deploys on push
# Cost: $7-25/mo
```

---

### **3. Hugging Face Spaces (Free tier!)**

```python
# app.py
import gradio as gr
import openai

def chat(message, history):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': message}]
    )
    return response['choices'][0]['message']['content']

demo = gr.ChatInterface(chat)
demo.launch()
```

**Deploy:** Push to Hugging Face repo → Auto-deployed!

---

## 🎯 Decision Tree

```
Need enterprise compliance?
├─ Yes → Azure OpenAI Service
└─ No → Continue

Budget < $100/mo?
├─ Yes → Railway/Render (managed)
└─ No → Continue

High volume (>100k requests/day)?
├─ Yes → Self-hosted on EC2/GCP Compute
└─ No → Serverless (Lambda/Cloud Run)

Need custom models?
├─ Yes → EC2 with GPU
└─ No → OpenAI API + Lambda
```

---

## 📊 Latency Comparison

| Setup | Cold Start | Avg Response | p99 |
|-------|-----------|--------------|-----|
| **Lambda + OpenAI** | 2-5s | 1.5s | 4s |
| **Cloud Run + OpenAI** | 1-3s | 1.2s | 3s |
| **EC2 + Self-hosted LLM** | 0s | 2-3s | 6s |
| **Railway** | 0s | 1.8s | 5s |

**Fastest:** Cloud Run (minimal cold starts)

---

## 🔧 Troubleshooting

### **High API Costs**
```python
# Add caching
import redis
cache = redis.Redis(host='your-redis', port=6379)

def get_cached_response(query):
    cached = cache.get(query)
    if cached:
        return cached.decode()

    response = openai.ChatCompletion.create(...)
    cache.setex(query, 3600, response)  # Cache for 1 hour
    return response
```

### **Slow Responses**
```python
# Use async
from fastapi import FastAPI
import asyncio

@app.post('/query')
async def query(request: QueryRequest):
    # Retrieve and generate in parallel
    retrieval_task = asyncio.create_task(retrieve(request.query))

    retrieved = await retrieval_task
    answer = await generate(request.query, retrieved)

    return {'answer': answer}
```

### **Out of Memory**
```python
# Clear CUDA cache (for self-hosted models)
import torch

torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

---

## 🎓 Next Steps

**Week 1:** Deploy to Railway/Render (get something live fast)
**Week 2:** Add monitoring and logging
**Week 3:** Optimize costs (caching, smaller models)
**Week 4:** Migrate to AWS/GCP if needed for scale

---

## 💡 Pro Tips

1. **Start small** - Railway/Render first, migrate to AWS later if needed
2. **Monitor everything** - You can't optimize what you don't measure
3. **Use managed services** - Don't run your own vector DB unless you have to
4. **Cache aggressively** - 80% of queries are similar
5. **Set billing alerts** - Prevent surprise $1000 bills

---

**Questions? Issues?**

Check platform docs:
- AWS: https://docs.aws.amazon.com/lambda
- GCP: https://cloud.google.com/vertex-ai/docs
- Azure: https://learn.microsoft.com/azure/ai-services/openai

---

**Made with ❤️ for AI builders**

*Deploy with confidence!*
