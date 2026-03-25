import unittest

from tokenizer.bpe import BYTE_VOCAB_SIZE
from tokenizer.bpe import split_text
from tokenizer.bpe import train_bpe


class BPETests(unittest.TestCase):
    def test_split_text_preserves_round_trip(self) -> None:
        text = "hi\n\na  b\n"
        rebuilt = b"".join(split_text(text)).decode("utf-8")
        self.assertEqual(rebuilt, text)

    def test_train_bpe_raises_for_vocab_below_byte_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            rf"vocab_size must be at least {BYTE_VOCAB_SIZE} for byte-level BPE\.",
        ):
            _ = train_bpe("hello", BYTE_VOCAB_SIZE - 1)

    def test_train_bpe_empty_text_returns_base_vocab(self) -> None:
        model = train_bpe("", BYTE_VOCAB_SIZE + 10)
        self.assertEqual(model.vocab_size, BYTE_VOCAB_SIZE)
        self.assertEqual(model.merges, ())
        self.assertEqual(model.encode(""), [])
        self.assertEqual(model.decode([]), "")

    def test_encode_decode_round_trip(self) -> None:
        text = "banana bandana\n"
        model = train_bpe(text, BYTE_VOCAB_SIZE + 4)

        token_ids = model.encode(text)

        self.assertTrue(any(token_id >= BYTE_VOCAB_SIZE for token_id in token_ids))
        self.assertEqual(model.decode(token_ids), text)

    def test_training_tie_break_is_deterministic(self) -> None:
        model = train_bpe("ab ac", BYTE_VOCAB_SIZE + 1)

        pair, token_id = model.merges[0]

        self.assertEqual(pair, (32, 97))
        self.assertEqual(model.vocab[token_id], b" a")


if __name__ == "__main__":
    unittest.main()
