import unittest

import bt


class ApiSurfaceTests(unittest.TestCase):
    def test_public_exports(self) -> None:
        self.assertTrue(hasattr(bt, "Tensor"))
        self.assertTrue(hasattr(bt, "full"))
        self.assertTrue(hasattr(bt, "zeros"))
        self.assertTrue(hasattr(bt, "ones"))
        self.assertTrue(hasattr(bt, "tensor"))


if __name__ == "__main__":
    unittest.main()
