/**
 * peachbase_bm25.h
 *
 * Optimized BM25 scoring for PeachBase.
 */

#ifndef PEACHBASE_BM25_H
#define PEACHBASE_BM25_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute BM25 score for a single document.
 *
 * BM25 formula:
 * score = sum over query terms of: IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
 *
 * @param term_freqs Array of term frequencies in the document (parallel to idfs)
 * @param idfs Array of IDF scores for query terms
 * @param n_terms Number of query terms
 * @param doc_len Document length (number of tokens)
 * @param avg_doc_len Average document length in corpus
 * @param k1 BM25 parameter (typically 1.5)
 * @param b BM25 parameter (typically 0.75)
 * @return BM25 score
 */
float peachbase_bm25_score(
    const int* term_freqs,
    const float* idfs,
    size_t n_terms,
    int doc_len,
    float avg_doc_len,
    float k1,
    float b
);

/**
 * Batch compute BM25 scores for multiple documents.
 *
 * @param term_freqs_matrix Matrix of term frequencies [n_docs x n_terms]
 * @param idfs Array of IDF scores for query terms
 * @param doc_lens Array of document lengths
 * @param n_docs Number of documents
 * @param n_terms Number of query terms
 * @param avg_doc_len Average document length
 * @param k1 BM25 parameter
 * @param b BM25 parameter
 * @param results Output array for scores (must be pre-allocated, size n_docs)
 */
void peachbase_batch_bm25_score(
    const int* term_freqs_matrix,
    const float* idfs,
    const int* doc_lens,
    size_t n_docs,
    size_t n_terms,
    float avg_doc_len,
    float k1,
    float b,
    float* results
);

#ifdef __cplusplus
}
#endif

#endif /* PEACHBASE_BM25_H */
