import os
import re
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# 1. LLM
# =========================
class CallLLM:
    def __init__(self, llmapi):
        self.url = llmapi['url']
        self.headers = {
            "Authorization": f"Bearer {llmapi['token']}",
            "Content-Type": "application/json"
        }
        self.model = llmapi['name']

    def generate(self, system, user):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0,
            "max_tokens": 1500
        }

        try:
            res = requests.post(self.url, json=payload, headers=self.headers)
            res.raise_for_status() # Catches 401 Unauthorized, 429 Too Many Requests, etc.
            data = res.json()

            # Safely navigate the JSON structure
            if data.get("choices") and len(data["choices"]) > 0:
                # Use .get() to prevent KeyError if 'message' or 'content' is missing
                text = data["choices"][0].get("message", {}).get("content")
                
                # Critical Fix: Verify text is actually a string before regex
                if isinstance(text, str) and text.strip():
                    return self.extract_sql(text)
                else:
                    print(f"❌ API GENERATION ERROR: 'content' was null or empty. Raw API data: {data}")
                    return "LLM Error: Empty generation"

            # If 'choices' is missing or empty, print the raw data for debugging
            print(f"❌ API RESPONSE ERROR: Unexpected JSON structure. Raw API data: {data}")
            return "LLM Error: Invalid response format"

        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP REQUEST ERROR: {e}")
            return "LLM Error: Connection failed"

    def extract_sql(self, text):
        # Defensive check just in case
        if not isinstance(text, str):
            return "LLM Error: Regex expected string"
            
        match = re.search(r"SELECT .*?;", text, re.I | re.S)
        return match.group(0) if match else text

# =========================
# 2. SIM
# =========================
class SemanticInputMasker:
    def __init__(self):
        self.term_repo = {
            "Rob Dinning": "Store_Name",
            "CDU-I": "UNIT_ALIAS"
        }

        self.meta_repo = {
            "Store_Name": "store name",
            "UNIT_ALIAS": "refinery unit"
        }

    def process(self, question):
        masked_q = question
        hints = []

        for term, field in self.term_repo.items():
            if term.lower() in question.lower():
                meta = self.meta_repo.get(field, field)
                masked_q = re.sub(term, f"[{meta}]", masked_q, flags=re.I)
                hints.append(f"{term} is value of {field}")

        return masked_q, " ".join(hints)


# =========================
# 3. CSR
# =========================
class ContextualSQLRetriever:
    def __init__(self, embedder):
        self.embedder = embedder

        self.qs_repo = [
            {
                "raw_q": "Did Blake book Adan Dinning?",
                "masked_q": "Did [customer name] book [store name]?",
                "sql": "SELECT * FROM bookings"
            }
        ]

        self.qs_embeddings = embedder.encode(
            [x["masked_q"] for x in self.qs_repo],
            normalize_embeddings=True
        )

    def retrieve(self, masked_q, threshold=0.8):
        q_emb = self.embedder.encode([masked_q], normalize_embeddings=True)[0]
        scores = self.qs_embeddings @ q_emb

        idx = np.argmax(scores)

        if scores[idx] > threshold:
            return self.qs_repo[idx]

        return None


# =========================
# 4. APC
# =========================
class AdaptivePromptComposer:
    def __init__(self, schema):
        self.schema = schema

    def build(self, question, hints, matched):
        if matched:
            system = "You are an expert SQL engineer."

            user = f"""
Example:
Q: {matched['raw_q']}
SQL: {matched['sql']}

Question:
{question}

Hints:
{hints}

Rules:
- Output ONLY SQL
"""
        else:
            schema_str = "\n".join(self.schema)

            system = "You are an expert SQL engineer."

            user = f"""
Schema:
{schema_str}

Question:
{question}

Hints:
{hints}

Rules:
- Output ONLY SQL
"""

        return system, user


# =========================
# 5. FULL PIPELINE
# =========================
class ADEPTPipeline:
    def __init__(self, llmapi, schema):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("moka-ai/m3e-base")
        print("Embedding loaded!")

        self.sim = SemanticInputMasker()
        self.csr = ContextualSQLRetriever(self.embedder)
        self.apc = AdaptivePromptComposer(schema)
        self.llm = CallLLM(llmapi)

    def run(self, question):
        print("\n=== PIPELINE START ===")

        # 1. SIM
        masked_q, hints = self.sim.process(question)
        print("[Masked]:", masked_q)
        print("[Hints]:", hints)

        # 2. CSR
        matched = self.csr.retrieve(masked_q)
        print("[Mode]:", "Few-shot" if matched else "Zero-shot")

        # 3. Prompt
        system, user = self.apc.build(question, hints, matched)

        # 4. LLM
        sql = self.llm.generate(system, user)

        print("[SQL]:", sql)
        return sql


# =========================
# 6. MAIN
# =========================
if __name__ == "__main__":

    llmapi = {
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'token': os.getenv("OPENROUTER_API_KEY", ""),
        'name': 'qwen/qwen3-14b'
    }

    schema = [
        "Table users(id, name, email)",
        "Table bookings(Booking_ID, Customer_ID, Flight_ID)"
    ]

    pipeline = ADEPTPipeline(llmapi, schema)

    question = "Hiển_thị tên người lãnh_đạo và địa_điểm của các trường đại_học ở Việt_Nam?"

    pipeline.run(question)