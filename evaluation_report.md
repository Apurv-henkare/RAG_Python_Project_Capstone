# RAG Capstone Evaluation Report

This report evaluates the performance of the **Love2D Documentation RAG Assistant** across 10 benchmark queries measuring retrieval capability, precision, query recall, answer accuracy, guardrail safety, and latency.

---

## 1. Overall Metrics Summary

| Metric | Score | In Simple Terms |
| :--- | :---: | :--- |
| **Query Recall (Hit Rate@5)** | **87.5%** (7/8) | Found the right PDF in the top 5 chunks for 7 out of 8 questions. |
| **Precision@5** | **65.0%** | Purity metric: On average, 65.0% of retrieved chunks came directly from the target PDF. |
| **Answer Accuracy (Key Facts)** | **100.0%** (8/8) | Successfully included required technical concepts in all 8 in-scope answers. |
| **Guardrail Accuracy (No Hallucination)** | **100.0%** (2/2) | Correctly rejected out-of-scope questions without making things up. |
| **Average Response Time** | **3.97 seconds** | Average end-to-end time from query to full answer. |

---

## 2. What Each Metric Means (In Simple Language)

### Metric 1: Query Recall (Hit Rate@5) — **87.5%**
- **What it measures**: Did FAISS manage to catch the right document in the top 5, or did it miss it completely?
- **Why it matters**: Proves whether the search engine finds the right manual when asked a question.

### Metric 2: Precision@5 — **65.0%**
- **What it measures**: Out of the 5 chunks FAISS pulled, how many were pure signal from the target PDF vs useless background noise?
- **Why it matters**: Higher precision means less junk sent to the LLM and lower token costs.

### Metric 3: Answer Accuracy (Key Concepts) — **100.0%**
- **What it measures**: Did the generated answer contain the required technical facts and code?

### Metric 4: Guardrail / Refusal Accuracy — **100.0%**
- **What it measures**: Did the model politely refuse when asked about out-of-scope topics like C# or physics without hallucinating?

### Metric 5: Average Response Time — **3.97 seconds**
- **What it measures**: Speed of retrieval (<0.05s) + Gemini generation (~3.5s).

---

## 3. Detailed Question-by-Question Results

| ID | Question | Expected PDF | Precision@5 | Query Recall (Hit@5) | Answer | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | What is Love2D and what language does it use? | `Love2d.pdf` | 60.0% (3/5) | **PASS** | **PASS** | 3.81 |
| **2** | How do you check for collisions between two entities in Lua? | `Collisions.pdf` | 0.0% (0/5)\* | **FAIL\*** | **PASS** | 3.93 |
| **3** | How can you play audio and sound effects in a game? | `Audio.pdf` | 60.0% (3/5) | **PASS** | **PASS** | 3.34 |
| **4** | How do you animate sprites and handle animation frames? | `Animation.pdf` | 100.0% (5/5) | **PASS** | **PASS** | 6.90 |
| **5** | How does a camera follow the player in Love2D? | `Camera.pdf` | 60.0% (3/5) | **PASS** | **PASS** | 3.38 |
| **6** | How do you implement jumping and gravity in a platformer? | `Platformer.pdf` | 100.0% (5/5) | **PASS** | **PASS** | 5.22 |
| **7** | How do tables work in Lua programming? | `Tables.pdf` | 80.0% (4/5) | **PASS** | **PASS** | 3.61 |
| **8** | How can you debug code and print errors in Love2D? | `Debugging.pdf` | 60.0% (3/5) | **PASS** | **PASS** | 4.18 |
| **9** | How do you write a Web API using C# and ASP.NET? | *Out of Scope* | N/A | **PASS** | **PASS** | 1.84 |
| **10** | Explain quantum physics equations for black holes. | *Out of Scope* | N/A | **PASS** | **PASS** | 3.48 |

*\*Note on Question 2: FAISS retrieved `Resolving_Collision.pdf` instead of `Collisions.pdf`. Both contain collision logic, which is why the answer was 100% correct.*

---

## 4. Key Insight: Precision and Query Recall in RAG

- Setting `K = 5` achieved an optimal balance: high Query Recall (87.5%), strong Precision (65.0%), and 100% Answer Accuracy with fast response times.
