import unittest
import numpy as np

from sentence_transformers import SentenceTransformer


class TestSentenceEmbeddings(unittest.TestCase):
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_is_deterministic(self) -> None:
        model = SentenceTransformer(self.MODEL_NAME)
        text = "Tim Burton collaborated with Johnny Depp on multiple films."
        emb1 = model.encode(text, normalize_embeddings=True)
        emb2 = model.encode(text, normalize_embeddings=True)
        self.assertEqual(emb1.shape, emb2.shape)
        cosine = float(np.dot(emb1, emb2))
        self.assertGreater(cosine, 0.999, msg=f"cosine too low: {cosine}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
