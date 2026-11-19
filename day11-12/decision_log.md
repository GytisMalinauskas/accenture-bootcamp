# Decision Log: Local RAG Pipeline

## Machine
- **Hardware:**
    - **CPU:** 12th Gen Intel Core i7-12800H (14 cores / 20 threads)
    - **RAM:** 31.7 GB (32 GB)
    - **GPU:** NVIDIA RTX A2000 8GB Laptop GPU
    - Ollama runs native Windows 100% GPU as show below.

- **`ollama ps` with my main model loaded:** 
``` 
NAME                 ID              SIZE      PROCESSOR    CONTEXT    RUNNER      UNTIL              
qwen3.5:4b-q4_K_M    d8b0f5e9760c    3.4 GB    100% GPU     8192       llamacpp    4 minutes from now    
```

## Model Selection
- **Chosen LLM:** qwen3.5:4b-q4_K_M
- **Thinking:** off, because it slows down the process and it is unnecessary.
- **Why:** I compared 2 models (`qwen3.5:2b-q4_K_M` vs `qwen3.5:4b-q4_K_M`) using the same prompts.

| Model | Avg Quality Rating | Ratings | Avg Speed | Avg Time | Total Completion Tokens |
| --- | --- | --- | --- | --- | --- |
| `qwen3.5:4b-q4_K_M` | 4.67 / 5 | `[5, 4, 5, 5, 5, 4]` | `36.5 tok/s` | `4.67 s` | `1,088` |
| `qwen3.5:2b-q4_K_M` | 4.33 / 5 | `[4, 4, 3, 5, 5, 5]` | `74.0 tok/s` | `2.76 s` | `1,320` |

The comparison was done by rating average response quality, average speed (tokens per second, tok/s), average time (seconds, s) and total completion tokens. I chose `qwen3.5:4b-q4_K_M` because its average quality rating was higher even though it was subjective and it generated less completion tokens. While `qwen3.5:2b-q4_K_M` proved to be faster, for this task quality is more significant because there aren't lots of data. [See the benchmark results by clicking here.](./benchmark_results.json)
- **What I considered but rejected:** As mentioned above I rejected the `qwen3.5:2b-q4_K_M` model because it was performed slightly worse than the bigger model(`qwen3.5:4b-q4_K_M`).

## Embedding Model
- **Chosen:** nomic-embed-text, with 768 embedding dimensions.
- **Why:** I compared nomic with mxbai-embed-large. Since model is large and it literally gives the same chunks as top-k, I chose nomic-embed-text.
#### Embedding model comparison.
```
NAME                        ID              SIZE      PROCESSOR    CONTEXT    RUNNER      UNTIL                   
mxbai-embed-large:latest    468836162de7    624 MB    100% GPU     512        llamacpp    About a minute from now    
nomic-embed-text:latest     0a109f422b47    335 MB    100% GPU     2048       llamacpp    About a minute from now 
```
## Parsing
- **Extractor:** pypdf with extraction_mode="layout".
#### Page with a Table after extraction and reference link below:
[See product_guide.pdf 2nd page](/day11-12/documents/product_guide.pdf) 
```
integrated platform approach ensures that the data generated during project
execution flows seamlessly into analytics — without manual exports,
transformations, or reconciliation.

Key Company Facts


 Metric                                           Value

 Founded                                             2017

 Headquarters                                     Munich, Germany

 Employees                                         ~400

 Customers                                         847 (as of Q1 2026)

 Countries served                                24

 Annual recurring revenue (ARR)        €18.2M (Q1 2026)

 Data centers                                      Frankfurt, Dublin, US-East (Virginia), Singapore

 SOC 2 Type II certified                        Since 2020

 ISO 27001 certified                            Since 2021


NovaTech processes over 12 billion API requests per month across its platform
and stores more than 48 petabytes of customer project and analytics data. The
platform is built on a microservices architecture deployed on Kubernetes, with
PostgreSQL and ClickHouse as the primary data stores.


2. NovaTech ProjectHub

2.1 Overview

NovaTech ProjectHub is a comprehensive project management platform designed
for mid-size to large organizations managing complex, cross-functional initiatives.
It supports Agile, Waterfall, and hybrid methodologies out of the box, with
configurable workflows that adapt to any team's processes.

ProjectHub is used by over 1,800 organizations, with the average enterprise
customer managing 340 active projects and 2,100 users on the platform
simultaneously.
```
## Chunking Strategy
- **Chunk size:** 1024 characters
- **Overlap:** 150 characters
- **Why:**
1. Experimented with 1024 chunk size and 150 characters overlap which produced 231 chunks.
2. Experimented with 512 chunk size and 50 characters overlap which produced 465 chunks.

| File (<span style="color:red">chunk size, overlap,</span> top-k) | `finish_reason` | `prompt_tokens` (min / max / avg) | Avg Gen Time | Answerable Misses | Unanswerable (`q17`, `q18`) |
| --- | --- | --- | --- | --- | --- |
| `pipeline_outputs_512c_50olp_k5.json` | all `stop` | `496 / 734 / 601.7` | `5931.7 ms` | 4 (`q05, q06, q10, q13`) | Both refused |
| `pipeline_outputs_512c_50olp_k8.json` | `stop`, `length` | `749 / 978 / 882.7` | `7413.3 ms` | 2 (`q05, q10`) | Both refused |
| `pipeline_outputs_1024c_150olp_k5.json` | all `stop` | `810 / 1243 / 1014.9` | `6237.0 ms` | 2 (`q05, q06`) | Both refused |
| `pipeline_outputs_1024c_150olp_k8.json` | all `stop` | `1352 / 1845 / 1525.2` | `6764.6 ms` | 1 (`q06` only) | Both refused |

Experiment showed that increasing chunk and overlap sizes gives the model more context which leads to a better result, less answerables missed but the tradeoff is more prompt tokens and more time spent.

## Retrieval Configuration
- **Top-K:** 8
- **Context window:** 
```
NAME                 ID              SIZE      PROCESSOR    CONTEXT    RUNNER      UNTIL              
qwen3.5:4b-q4_K_M    d8b0f5e9760c    3.3 GB    100% GPU     8192       llamacpp    4 minutes from now 
```
- **Why:**
1. Experimented with 5 top-k.
2. Experimented with 8 top-k.

| File (chunk size, overlap, <span style="color:red">top-k</span>) | `finish_reason` | `prompt_tokens` (min / max / avg) | Avg Gen Time | Answerable Misses | Unanswerable (`q17`, `q18`) |
| --- | --- | --- | --- | --- | --- |
| `pipeline_outputs_512c_50olp_k5.json` | all `stop` | `496 / 734 / 601.7` | `5931.7 ms` | 4 (`q05, q06, q10, q13`) | Both refused |
| `pipeline_outputs_512c_50olp_k8.json` | `stop`, `length` | `749 / 978 / 882.7` | `7413.3 ms` | 2 (`q05, q10`) | Both refused |
| `pipeline_outputs_1024c_150olp_k5.json` | all `stop` | `810 / 1243 / 1014.9` | `6237.0 ms` | 2 (`q05, q06`) | Both refused |
| `pipeline_outputs_1024c_150olp_k8.json` | all `stop` | `1352 / 1845 / 1525.2` | `6764.6 ms` | 1 (`q06` only) | Both refused |

Experiment showed that increasing top-k size gives more context to the model which allows the model to answer more questions with posibly more accurate results.

## Observations
- **What worked well:** q17 and q18 both unanswerable questions were refused to answer. 
- **What failed:** Failed q06 (the answer not found in given context) and q10 (only 2 out of 3 board requests retrieved). In total 16/18 tests passed. 
- **Traps I hit:** Since I decided to go with `nomic-embed-text` embedding model which has 768 dimensions, the last ingestion was run by `mxbai-embed-large` which has 1024 dimensions and when I tried to run the pipeline.py an error occured because of the incompatibility.
- **Local vs cloud expectations:** I think that local models are faster but the quality is significantly better with the cloud models.

## If I Had More Time / Better Hardware
- I would try to find the best hyperparameters and experiment with more powerful local models.
