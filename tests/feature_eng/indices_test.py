from retailsynth.feature_eng.indices import IndexMap


class TestIndices:
    im = IndexMap([1, 2, 3], name="customer_key")

    def test_get_index(self):
        assert self.im.get_index(1) == [0]
        assert self.im.get_index([1, 2]) == [0, 1]

    def test_get_values(self):
        assert self.im.get_values(0) == [1]
        assert self.im.get_values([0, 1]) == [1, 2]
