import unittest


class TestNanobindAdd(unittest.TestCase):
    def test_add(self) -> None:
        import bt

        self.assertEqual(bt.add(1, 2), 3)
        self.assertEqual(bt.add(a=1, b=2), 3)
        self.assertEqual(bt.add(10), 11)


if __name__ == "__main__":
    unittest.main()
