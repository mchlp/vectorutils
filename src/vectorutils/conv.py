"""Collection of classes for converting between vector formats"""

import numpy as np
import pyarrow as pa
import faiss
from pyserini.index import lucene


class FaissConverter:
    """Class for converting from faiss vector format"""
    def __init__(self, index: faiss.Index):
        self.index = index

    def to_arrow(self):
        """Returns an arrow table containing the vectors"""                 
        vectors_np = self.to_numpy()
        vectors_np_flat = vectors_np.reshape(-1)
        vectors_arrow = pa.FixedSizeListArray.from_arrays(vectors_np_flat, self.index.d)
        return vectors_arrow
    
    def to_numpy(self):
        """Returns a numpy array containing the vectors"""
        num_vectors = self.index.ntotal
        vector_dimension = self.index.d
        vectors_np = np.empty((num_vectors, vector_dimension), dtype=np.float32)
        
        self.index.reconstruct_n(0, num_vectors, vectors_np)
        return vectors_np
