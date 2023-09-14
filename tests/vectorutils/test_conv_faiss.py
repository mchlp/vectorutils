"""Tests for converting from faiss format"""

import numpy as np
import faiss
import pyarrow as pa
import pytest

from vectorutils import conv


@pytest.fixture(name="faiss_index")
def faiss_index_fixture():
    """Fixture for generating a static 3x3 faiss index for tests"""
    dataset = np.array([[1.1,2.1,3.1], [4.2,5.2,6.2], [7.3,8.3,9.3]], dtype=np.float32)
    index = faiss.index_factory(3, "Flat")
    index.add(dataset)
    return index


def test_conv_faiss_to_numpy(faiss_index):
    """Tests that conversion from faiss to numpy works"""
    converter = conv.FaissConverter(faiss_index)
    numpy_vec = converter.to_numpy()
    assert np.array_equal(numpy_vec, np.array([[1.1,2.1,3.1], [4.2,5.2,6.2], [7.3,8.3,9.3]], dtype=np.float32))


def test_conv_faiss_to_arrow(faiss_index):
    """Tests that conversion from faiss to arrow works"""
    converter = conv.FaissConverter(faiss_index)
    arrow_vec = converter.to_arrow()
    expected_vec = pa.FixedSizeListArray.from_arrays([1.1,2.1,3.1,4.2,5.2,6.2,7.3,8.3,9.3], 3).cast(pa.list_(pa.float32(), 3))
    assert arrow_vec.equals(expected_vec)
