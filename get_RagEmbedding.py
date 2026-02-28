#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import time
from tqdm import tqdm

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-query", "--query_path", type=str, required=True, help="Path to the query .npy file")
parser.add_argument("-db", "--database_path", type=str, required=True, help="Path to the RAG database .npy file")
parser.add_argument("-out", "--output_path", type=str, required=True, help="Path to save the fused embeddings")
parser.add_argument("-batch", "--batch_size", type=int, default=100, help="Number of query sequences per batch")
args = parser.parse_args()

# Parameters
query_path = args.query_path
database_path = args.database_path
output_path = args.output_path
batch_size = args.batch_size

# Fixed parameters
maxseq = 15
num_feature = 1024  # Embedding dimension
dim = maxseq * num_feature  # Flattened dimension: 15 * 1024 = 15360

# Load Query and RAG Database
query_data = np.load(query_path)  # Shape: (N, 1, 15, 1024)
rag_database = np.load(database_path)  # Shape: (M, 1, 15, 1024)

# Flatten embeddings for efficient distance computation
query_flat = np.reshape(query_data, (query_data.shape[0], dim))  # (N_queries, dim)
rag_flat = np.reshape(rag_database, (rag_database.shape[0], dim))  # (M_db, dim)

# Precompute squared norms for RAG database (used in squared Euclidean distance)
rag_norms = np.sum(rag_flat ** 2, axis=1)  # (M_db,)

# Initialize fused embeddings as copy of query (flattened)
fused_flat = np.copy(query_flat)

# Process Query Data in Batches
num_queries = query_data.shape[0]
num_batches = (num_queries + batch_size - 1) // batch_size

print(f"Total queries: {num_queries}")
print("Fusing with ratio: Query 0.7 and RAG 0.3")
print(f"Processing in {num_batches} batches of size {batch_size}")

for batch_idx in tqdm(range(num_batches)):
    start_time = time.time()
    
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_queries)
    
    batch_queries_flat = query_flat[start_idx:end_idx]
    
    # Compute squared L2 distances vectorized over the batch
    q_norms = np.sum(batch_queries_flat ** 2, axis=1)[:, None]  # (batch_size, 1)
    dot_products = batch_queries_flat @ rag_flat.T  # (batch_size, M_db)
    dist_sq = q_norms + rag_norms[None, :] - 2 * dot_products  # (batch_size, M_db)
    
    # Get top-5 closest indices per query in the batch
    top5_indices = np.argsort(dist_sq, axis=1)[:, :5]  # (batch_size, 5)
    
    # Retrieve the top-5 embeddings, average them
    closest = rag_flat[top5_indices]  # (batch_size, 5, dim)
    avg_embeddings = np.mean(closest, axis=1)  # (batch_size, dim)
    
    # Fuse: 70% query + 30% average RAG
    fused_batch = batch_queries_flat * 0.7 + avg_embeddings * 0.3
    
    # Update fused_flat
    fused_flat[start_idx:end_idx] = fused_batch
    
    end_time = time.time()
    elapsed = end_time - start_time
    tqdm.write(f"Batch {batch_idx + 1}/{num_batches} processed in {elapsed:.2f} seconds")

# Reshape fused embeddings back to original shape
fused_embeddings = np.reshape(fused_flat, query_data.shape)  # (N_queries, 1, 15, 1024)

# Save the final fused embeddings
# np.save(output_path, fused_embeddings)
# Save compressed
output_path = output_path.replace(".npy", ".npz")  # optional but recommended
np.savez_compressed(output_path, fused_embeddings=fused_embeddings)
print(f"Fused embeddings saved to {output_path}")