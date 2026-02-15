import unittest


class TestNanobindDog(unittest.TestCase):
    def test_dog(self) -> None:
        import bt

        d = bt.Dog("Max")
        self.assertEqual(d.name, "Max")
        self.assertEqual(d.bark(), "Max: woof!")

        d.name = "Charlie"
        self.assertEqual(d.bark(), "Charlie: woof!")


if __name__ == "__main__":
    unittest.main()
