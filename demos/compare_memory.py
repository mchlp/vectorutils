import faiss
import util
from pympler import asizeof
from pyserini.index import lucene

from vectorutils import conv

FAISS_INDEX_PATH = "/u9/yzpu/.cache/pyserini/indexes/faiss.msmarco-v1-passage.tct_colbert-v2-hnp.20210608.5f341b.53bcaa78ab0ca629f3379b8aa00eb3ae/index"
LUCENE_INDEX_PATH = "/u9/yzpu/research/pyserini/indexes/lucene-index-msmarco-passage"


def convert_dense_vector_arrow():
    print("=== Converting FAISS (Dense) to Arrow ===")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with util.TimerLogger():
        converter = conv.FaissConverter(index)
        arrow_vector = converter.to_arrow()
    print(f"Size of arrow vectors: {asizeof.asizeof(arrow_vector)}")


def convert_dense_vector_numpy():
    print("=== Converting FAISS (Dense) to Numpy ===")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with util.TimerLogger():
        converter = conv.FaissConverter(index)
        numpy_vector = converter.to_numpy()
    print(f"Size of numpy vectors: {asizeof.asizeof(numpy_vector)}")


def convert_sparse_vector_arrow():
    print("=== Converting Lucene (Sparse) to Arrow ===")
    index = lucene.IndexReader(LUCENE_INDEX_PATH)
    with util.TimerLogger():
        converter = conv.FaissConverter(index)
        arrow_vector = converter.to_arrow()
    print(f"Size of arrow vectors: {asizeof.asizeof(arrow_vector)}")

convert_dense_vector_arrow()
convert_dense_vector_numpy()

# print("Convert to arrow")
# start = time.time()
# converter = conv.FaissConverter(index)
# arrow_vector = converter.to_arrow()
# end = time.time()
# print(f"Conversion took {end-start}s")
# print(f"Size of vector: {util.total_size(arrow_vector)}")

# print()
# print("Convert to numpy")
# start = time.time()
# converter = conv.FaissConverter(index)
# numpy_vector = converter.to_numpy()
# end = time.time()
# print(f"Conversion took {end-start}s")
# print(f"Size of vector: {util.total_size(numpy_vector)}")
# print(sys.getsizeof(numpy_vector))
# print(numpy_vector.nbytes)
# print(numpy_vector.size)
# print(numpy_vector.itemsize)

# print()
# print(f"Num vectors: {num_vectors}")
# print("Checking that FAISS vector and arrow vector are equal...")
# arrow_offset = 0
# num_values = 0

# for i in tqdm(range(num_vectors)):
#     faiss_vector = index.reconstruct(i)    
#     vector_len = len(faiss_vector)
#     for j in range(vector_len):
#         if abs(faiss_vector[j] - arrow_vector[arrow_offset].as_py()) > EPSILON:
#             print(f"Not equal for vector {i} at index {j}")
#             print(f"FAISS vector: {faiss_vector[j]}")
#             print(f"Arrow vector: {arrow_vector[arrow_offset]}")
#         arrow_offset += 1
#         num_values += 1
# print(f"Finished checking {num_values} values")
