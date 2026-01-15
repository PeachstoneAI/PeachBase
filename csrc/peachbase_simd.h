/**
 * peachbase_simd.h
 *
 * SIMD-optimized vector similarity operations for PeachBase.
 * Supports AVX2, AVX-512, and fallback implementations.
 */

#ifndef PEACHBASE_SIMD_H
#define PEACHBASE_SIMD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CPU feature detection flags
 */
typedef enum {
    PEACHBASE_CPU_FALLBACK = 0,
    PEACHBASE_CPU_AVX2 = 1,
    PEACHBASE_CPU_AVX512 = 2
} peachbase_cpu_features_t;

/**
 * Detect available CPU features at runtime.
 *
 * @return CPU feature level (FALLBACK, AVX2, or AVX512)
 */
peachbase_cpu_features_t peachbase_detect_cpu_features(void);

/**
 * Compute cosine similarity between two vectors using SIMD.
 *
 * @param vec1 First vector (float array)
 * @param vec2 Second vector (float array)
 * @param dim Vector dimension
 * @return Cosine similarity (between -1 and 1)
 */
float peachbase_cosine_similarity_simd(const float* vec1, const float* vec2, size_t dim);

/**
 * Compute L2 (Euclidean) distance between two vectors using SIMD.
 *
 * @param vec1 First vector (float array)
 * @param vec2 Second vector (float array)
 * @param dim Vector dimension
 * @return L2 distance (non-negative)
 */
float peachbase_l2_distance_simd(const float* vec1, const float* vec2, size_t dim);

/**
 * Compute dot product between two vectors using SIMD.
 *
 * @param vec1 First vector (float array)
 * @param vec2 Second vector (float array)
 * @param dim Vector dimension
 * @return Dot product
 */
float peachbase_dot_product_simd(const float* vec1, const float* vec2, size_t dim);

/**
 * Batch compute cosine similarities between query and multiple vectors.
 *
 * @param query Query vector (float array)
 * @param vectors Matrix of vectors (flattened, row-major)
 * @param n_vectors Number of vectors
 * @param dim Vector dimension
 * @param results Output array for results (must be pre-allocated, size n_vectors)
 */
void peachbase_batch_cosine_similarity(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
);

/**
 * Batch compute L2 distances between query and multiple vectors.
 *
 * @param query Query vector (float array)
 * @param vectors Matrix of vectors (flattened, row-major)
 * @param n_vectors Number of vectors
 * @param dim Vector dimension
 * @param results Output array for results (must be pre-allocated, size n_vectors)
 */
void peachbase_batch_l2_distance(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
);

/**
 * Batch compute dot products between query and multiple vectors.
 *
 * @param query Query vector (float array)
 * @param vectors Matrix of vectors (flattened, row-major)
 * @param n_vectors Number of vectors
 * @param dim Vector dimension
 * @param results Output array for results (must be pre-allocated, size n_vectors)
 */
void peachbase_batch_dot_product(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
);

#ifdef __cplusplus
}
#endif

#endif /* PEACHBASE_SIMD_H */
