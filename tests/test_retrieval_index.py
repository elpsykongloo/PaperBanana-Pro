import unittest

from utils.retrieval_index import BM25Field, ReferencePoolIndex, cjk_ratio, tokenize


class RetrievalIndexTest(unittest.TestCase):
    def test_tokenize_drops_stopwords_numbers_and_chinese(self):
        tokens = tokenize("The Figure 1: 一个 Transformer encoder with 12 layers, using RoPE.")
        self.assertEqual(tokens, ["transformer", "encoder", "layers", "rope"])

    def test_cjk_ratio(self):
        self.assertEqual(cjk_ratio("plain english"), 0.0)
        self.assertGreater(cjk_ratio("整体框架示意图 overview"), 0.3)
        self.assertLess(cjk_ratio("a long english caption about transformers 图"), 0.1)

    def test_bm25_prefers_rare_terms_and_overlap_counts_terms(self):
        field = BM25Field([["encoder", "decoder"], ["encoder", "diffusion"], ["encoder", "graph"]])
        scores = field.score(["encoder", "diffusion"])
        self.assertEqual(int(scores.argmax()), 1)
        self.assertEqual(list(field.overlap(["encoder", "graph", "graph"])), [1.0, 1.0, 2.0])

    def test_rank_returns_all_items_with_ties_in_file_order(self):
        items = [
            {"id": "a", "visual_intent": "bar chart of revenue", "content": {"year": [2020]}},
            {"id": "b", "visual_intent": "line chart of accuracy", "content": {"step": [1]}},
            {"id": "c", "visual_intent": "heatmap of attention", "content": {"head": [1]}},
        ]
        index = ReferencePoolIndex(items, caption_weight=1.0, overlap_weight=2.0)
        order, token_count = index.rank("A heatmap showing attention weights", {"head": [1, 2]})
        self.assertEqual(order[0], 2)
        self.assertEqual(sorted(order), [0, 1, 2])
        self.assertGreater(token_count, 0)
        empty_order, empty_count = index.rank("", "")
        self.assertEqual(empty_order, [0, 1, 2])
        self.assertEqual(empty_count, 0)


if __name__ == "__main__":
    unittest.main()
