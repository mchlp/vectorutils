import numpy as np
import pyarrow as pa


class FaissToArrowConverter:
    def __init__(self):
        self.indexes = []

    def add_index(self, index):
        self.indexes.append(index)

    def _convert_index_to_arrow(self, index):
        num_vectors = index.ntotal
        vector_dimension = index.d
        vectors_np = np.empty((num_vectors, vector_dimension), dtype=np.float32)

        # Iterate through the index and copy the vectors to the numpy array
        for i in range(num_vectors):
            index.reconstruct(i, vectors_np[i])

        # Reshape the numpy array to make each vector one-dimensional
        vectors_flat = vectors_np.reshape(-1)

        # Convert the flat numpy array to a PyArrow array
        vectors_arrow = pa.array(vectors_flat)

        # Create a PyArrow schema with a single field for the vectors
        vector_field = pa.field("vectors", pa.float32(), nullable=False)
        schema = pa.schema([vector_field])

        # Create a PyArrow table with the vectors array
        table = pa.Table.from_arrays([vectors_arrow], schema=schema)
        return table

    def convert_all_to_arrow(self):
        arrow_tables = []
        for index in self.indexes:
            arrow_table = self._convert_index_to_arrow(index)
            arrow_tables.append(arrow_table)
        return arrow_tables