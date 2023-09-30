import time
import faiss
import sys

from pympler import asizeof
from vectorutils import conv
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa


INDEX_PATH = "/u9/yzpu/.cache/pyserini/indexes/faiss.msmarco-v1-passage.tct_colbert-v2-hnp.20210608.5f341b.53bcaa78ab0ca629f3379b8aa00eb3ae/index"
EPSILON = 1e-9

index = faiss.read_index(INDEX_PATH)
num_vectors = index.ntotal
print(f"Size of index: {asizeof.asizeof(index)}")
print()

print("Convert to arrow")
start = time.time()
converter = conv.FaissConverter(index)
arrow_vector = converter.to_arrow()
end = time.time()
print(f"Conversion took {end-start}s")
print(f"Size of vector: {asizeof.asizeof(arrow_vector)}")

arrow_vector_sub = pa.RecordBatch.from_arrays([arrow_vector.slice(length=500000)], names=["data"])

with pa.OSFile('msmacro_vectors.arrow', 'wb') as sink:
    with pa.ipc.new_file(sink, schema=arrow_vector_sub.schema) as writer:
        writer.write(arrow_vector_sub)

print()
print("Convert to numpy")
start = time.time()
converter = conv.FaissConverter(index)
numpy_vector = converter.to_numpy()
end = time.time()
print(f"Conversion took {end-start}s")
print(f"Size of vector: {asizeof.asizeof(numpy_vector)}")
print(sys.getsizeof(numpy_vector))
print(numpy_vector.nbytes)
print(numpy_vector.size)
print(numpy_vector.itemsize)

print()
print(f"Num vectors: {num_vectors}")
print("Checking that FAISS vector and arrow vector are equal...")
arrow_offset = 0
num_values = 0

for i in tqdm(range(num_vectors)):
    faiss_vector = index.reconstruct(i)    
    vector_len = len(faiss_vector)
    for j in range(vector_len):
        if abs(faiss_vector[j] - arrow_vector[arrow_offset].as_py()) > EPSILON:
            print(f"Not equal for vector {i} at index {j}")
            print(f"FAISS vector: {faiss_vector[j]}")
            print(f"Arrow vector: {arrow_vector[arrow_offset]}")
        arrow_offset += 1
        num_values += 1
print(f"Finished checking {num_values} values")
