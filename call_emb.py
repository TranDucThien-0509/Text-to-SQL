from sentence_transformers import SentenceTransformer

class LocalEmbedding:
    def __init__(self, model_name="moka-ai/m3e-base"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded!")

    def encode(self, texts):
        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True
            )

            return embeddings.tolist()

        except Exception as e:
            print("Embedding error:", e)
            return "Emb Fail"


# ======================
# TEST
# ======================
if __name__ == "__main__":

    emb_model = LocalEmbedding()

    sentences = [
        "Hello world",
        "Xin chào thế giới",
        "Text to SQL pipeline"
    ]

    vectors = emb_model.encode(sentences)

    for s, v in zip(sentences, vectors):
        print("Sentence:", s)
        print("Dim:", len(v))
        print()