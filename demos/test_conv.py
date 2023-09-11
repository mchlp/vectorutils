import time
import faiss

from vectorutils import conv
from tqdm import tqdm


INDEX_PATH = "/u9/yzpu/.cache/pyserini/indexes/faiss.msmarco-v1-passage.tct_colbert-v2-hnp.20210608.5f341b.53bcaa78ab0ca629f3379b8aa00eb3ae/index"
EPSILON = 1e-9

index = faiss.read_index(INDEX_PATH)
num_vectors = index.ntotal

start = time.time()
converter = conv.FaissToArrowConverter()
converter.add_index(index)
arrow_table = converter.convert_all_to_arrow()
end = time.time()

print(f"Conversion took {end-start}s")

print(f"Num vectors: {num_vectors}")
# print("Checking that FAISS vector and arrow vector are equal...")
# arrow_offset = 0
# arrow_vector = arrow_table[0].column(0)
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