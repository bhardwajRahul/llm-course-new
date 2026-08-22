## What's the difference between Tokens and API keys?

We use API keys and tokens for authentication and authorization.

But they serve different purposes and have distinct characteristics.

Tokens (like JWT - JSON Web Tokens):

Carries user context and permissions for authentication and authorization.

Encoded with a user ID, permissions, and expiration time, often in JWT format.

Critical for user-specific access, like accessing a user's profile data in an e-commerce platform.

It is issued by an authentication server after user login and contains user-specific information.

API Key:

Primarily for identifying the application or the consumer making the API call.

They are long strings we pass in the header or as a query parameter in the API request.

You use API keys when access does not involve user context. For example, accessing a public API or service-to-service communication.

They are long-lived and created through the API provider's platform or admin console.


### You have deployed an LLM/RAG application to production. How would you design an automated evaluation and guardrail system to ensure answer quality and reliability?

1. Core Objectives

| Objective | Key Metrics | Tools/Techniques |
| --- | --- | --- |
| **Answer Quality** | Accuracy, relevance, faithfulness | Automated evals, human-in-the-loop |
| **Reliability** | Latency, uptime, consistency | Monitoring, retries, fallbacks |
| **Safety** | Toxicity, bias, PII leakage | Content moderation, filters |
| **User Alignment** | Helpfulness, satisfaction | User feedback, A/B testing |

  
    
Automated Evaluation System

A. Ground Truth & Benchmarking
Curated Test Sets:Maintain a golden dataset of known questions/answers (e.g., from past user queries or synthetic data).
Use RAG-specific benchmarks (e.g., BEIR, MTEB).

Dynamic Evaluation:Periodically sample live queries and compare LLM outputs against ground truth (if available) or consensus from multiple models.


B. Automated Metrics
Use programmatic evaluators to score responses on:
Faithfulness (RAG-specific):Context Recall: Does the answer cite the correct chunks from the retrieval step? (Use RAGAS or custom logic.)
Answer Correctness: Is the answer factually correct given the context? (Use LLM-as-a-judge or rule-based checks.)

Relevance:Semantic Similarity: Compare answer to query using embeddings (e.g., sentence-transformers).
User Intent Match: Use a fine-tuned classifier to check if the answer addresses the intent.

Fluency & Coherence:Perplexity: Score answer fluency with a small LM (e.g., distilbert).
Repetition/PII Checks: Detect hallucinations, contradictions, or nonsensical outputs.


C. Continuous Evaluation Pipeline
Scheduled Jobs:Run daily/weekly evals on sampled queries (e.g., 100–1000 queries/day).
Compare before/after deployments to catch regressions.

Real-Time Shadow Mode:Deploy a canary model alongside production, compare outputs, and flag discrepancies.

Anomaly Detection:Use statistical process control (e.g., moving averages) to detect drops in quality metrics.


3. Guardrail System
A. Pre-Response Guardrails

| Guardrail Type | Implementation | Example Tools/Libraries |
| --- | --- | --- |
| **Toxicity Filtering** | Classify input/output for toxicity | Hugging Face `detoxify`, Perspective API |
| **PII Detection** | Scan for emails, SSNs, phone numbers | `presidio`, `spaCy` + regex |
| **Prompt Injection** | Detect adversarial prompts | Custom rules, `instruct-guardrail` |
| **Bias Mitigation** | Flag biased language | Fairlearn, `textblob` sentiment |
| **Hallucination Check** | Compare answer to retrieved context | RAGAS, custom LLM-as-a-judge |


B. Post-Response Guardrails
User Feedback Loop:Explicit: Thumbs up/down, ratings (e.g., "Was this answer helpful?").
Implicit: Track click-through rates, time-to-next-query, or copy-paste actions.

Automated Retries:If confidence score < threshold, re-generate with a different prompt/model.

Fallback Mechanisms:If RAG fails, fall back to a simpler model or return a "no answer" template.

C. Real-Time Monitoring
Latency & Errors:Track p50/p95/p99 latency, retrieval failures, LLM timeouts.

Quality Metrics:Monitor faithfulness score, relevance score, user satisfaction in real-time.

Alerting:Set up Slack/PagerDuty alerts for:Sudden drops in quality metrics.
Spikes in toxicity/PII violations.
High latency or error rates.



4. Human-in-the-Loop (HITL)
Sampling for Review:Randomly sample 1–5% of queries for human review (e.g., via Label Studio or Prodigy).
Prioritize low-confidence answers (e.g., faithfulness < 0.6).

Active Learning:Use human feedback to fine-tune the model or improve retrieval.

Escalation Path:For high-stakes queries (e.g., medical/legal), route to human experts.


5. RAG-Specific Guardrails
A. Retrieval Quality Checks
Context Relevance:Use cross-encoder models (e.g., bge-reranker) to score retrieved chunks.
Reject answers if top-k chunks have low relevance scores.

Diversity:Ensure retrieved chunks are not all from the same source (avoid bias).

Freshness:For time-sensitive queries, check document timestamps and flag stale context.

B. Answer Generation Checks
Citation Validation:Verify that every claim in the answer is supported by the retrieved context.
Use LLM-as-a-judge to ask: "Is this answer fully supported by the context?"

Contradiction Detection:Check for internal contradictions in the answer (e.g., with nli models).


6. Deployment & CI/CD Integration
Pre-Deployment Checks:Run automated evals on a staging dataset before promoting to production.
Canary Deployments: Roll out to 5–10% of traffic, monitor metrics, then scale.

Post-Deployment:Automated rollback if quality metrics drop below thresholds.
A/B Testing: Compare new vs. old models on live traffic.

```
User Query
    │
    ▼
┌───────────────────────────────────────┐
│           Pre-Processing               │
│  - Toxicity/PII Check (Detoxify)       │
│  - Prompt Injection Detection          │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│           RAG Pipeline                 │
│  - Retrieval (FAISS/Weaviate)          │
│  - Re-ranking (Cross-Encoder)          │
│  - Context Relevance Check             │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│           LLM Generation               │
│  - Answer Generation (vLLM)           │
│  - Hallucination Check (RAGAS)         │
│  - Citation Validation                 │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│           Post-Processing              │
│  - Toxicity/PII Check (Presidio)       │
│  - User Feedback Collection            │
│  - Quality Metrics Logging             │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│           Monitoring & Alerts          │
│  - Prometheus/Grafana                  │
│  - PagerDuty/Slack Alerts               │
│  - Human Review Queue (Label Studio)   │
└───────────────────────────────────────┘

```

Tools & Libraries Summary

| Category | Tools/Libraries |
| --- | --- |
| **Evaluation** | RAGAS, TruLens, DeepEval, Helm |
| **Guardrails** | Detoxify, Presidio, Guardrails AI, LangSmith |
| **Monitoring** | Prometheus, Grafana, ELK, OpenTelemetry |
| **RAG Optimization** | FAISS, Weaviate, Sentence Transformers |
| **LLM Inference** | vLLM, TensorRT-LLM, ONNX Runtime |
| **Human Review** | Label Studio, Prodigy, Scale AI |
| **CI/CD** | GitHub Actions, GitLab CI, Jenkins |

Key Trade-offs

| Approach | Pros | Cons |
| --- | --- | --- |
| **Automated Evals** | Scalable, fast | May miss nuanced issues |
| **Human Review** | High accuracy | Slow, expensive |
| **Shadow Mode** | No user impact | Requires duplicate resources |
| **Rule-Based Checks** | Fast, interpretable | Brittle, hard to maintain |
| **LLM-as-a-Judge** | Flexible, adaptive | Computationally expensive |
    




## Links:
- [Start Building These Projects to Become an LLM Engineer](https://dswharshit.medium.com/start-building-these-projects-to-become-an-llm-engineer-0064e9e68d9d)
