/**
 * peachbase_bm25.c
 *
 * Optimized BM25 scoring implementation for PeachBase.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "peachbase_bm25.h"
#include <math.h>

/**
 * Compute BM25 score for a single document.
 */
float peachbase_bm25_score(
    const int* term_freqs,
    const float* idfs,
    size_t n_terms,
    int doc_len,
    float avg_doc_len,
    float k1,
    float b
) {
    float score = 0.0f;
    float normalized_len = 1.0f - b + b * ((float)doc_len / avg_doc_len);

    for (size_t i = 0; i < n_terms; i++) {
        int tf = term_freqs[i];
        if (tf > 0) {
            float idf = idfs[i];
            float numerator = (float)tf * (k1 + 1.0f);
            float denominator = (float)tf + k1 * normalized_len;
            score += idf * (numerator / denominator);
        }
    }

    return score;
}

/**
 * Batch compute BM25 scores for multiple documents.
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
) {
    for (size_t doc_idx = 0; doc_idx < n_docs; doc_idx++) {
        const int* term_freqs = &term_freqs_matrix[doc_idx * n_terms];
        int doc_len = doc_lens[doc_idx];

        results[doc_idx] = peachbase_bm25_score(
            term_freqs,
            idfs,
            n_terms,
            doc_len,
            avg_doc_len,
            k1,
            b
        );
    }
}

/* ========================================================================
 * Python bindings
 * ======================================================================== */

static PyObject* py_bm25_score(PyObject* self, PyObject* args) {
    PyObject *term_freqs_obj, *idfs_obj;
    int doc_len;
    float avg_doc_len, k1, b;

    if (!PyArg_ParseTuple(args, "OOiff", &term_freqs_obj, &idfs_obj,
                          &doc_len, &avg_doc_len, &k1, &b)) {
        return NULL;
    }

    // Get buffers
    Py_buffer tf_buf, idf_buf;
    if (PyObject_GetBuffer(term_freqs_obj, &tf_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(idfs_obj, &idf_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&tf_buf);
        return NULL;
    }

    size_t n_terms_tf = tf_buf.len / sizeof(int);
    size_t n_terms_idf = idf_buf.len / sizeof(float);

    if (n_terms_tf != n_terms_idf) {
        PyBuffer_Release(&tf_buf);
        PyBuffer_Release(&idf_buf);
        PyErr_SetString(PyExc_ValueError, "term_freqs and idfs must have same length");
        return NULL;
    }

    float score = peachbase_bm25_score(
        (const int*)tf_buf.buf,
        (const float*)idf_buf.buf,
        n_terms_tf,
        doc_len,
        avg_doc_len,
        k1,
        b
    );

    PyBuffer_Release(&tf_buf);
    PyBuffer_Release(&idf_buf);

    return PyFloat_FromDouble((double)score);
}

static PyObject* py_batch_bm25_score(PyObject* self, PyObject* args) {
    PyObject *term_freqs_matrix_obj, *idfs_obj, *doc_lens_obj;
    float avg_doc_len, k1, b;

    if (!PyArg_ParseTuple(args, "OOOfff", &term_freqs_matrix_obj, &idfs_obj,
                          &doc_lens_obj, &avg_doc_len, &k1, &b)) {
        return NULL;
    }

    // Get buffers
    Py_buffer tf_buf, idf_buf, dl_buf;
    if (PyObject_GetBuffer(term_freqs_matrix_obj, &tf_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(idfs_obj, &idf_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&tf_buf);
        return NULL;
    }
    if (PyObject_GetBuffer(doc_lens_obj, &dl_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&tf_buf);
        PyBuffer_Release(&idf_buf);
        return NULL;
    }

    size_t n_terms = idf_buf.len / sizeof(float);
    size_t n_docs = dl_buf.len / sizeof(int);
    size_t expected_tf_size = n_docs * n_terms * sizeof(int);

    if (tf_buf.len != expected_tf_size) {
        PyBuffer_Release(&tf_buf);
        PyBuffer_Release(&idf_buf);
        PyBuffer_Release(&dl_buf);
        PyErr_SetString(PyExc_ValueError, "term_freqs_matrix dimensions mismatch");
        return NULL;
    }

    // Allocate results
    float* results = (float*)malloc(n_docs * sizeof(float));
    if (!results) {
        PyBuffer_Release(&tf_buf);
        PyBuffer_Release(&idf_buf);
        PyBuffer_Release(&dl_buf);
        return PyErr_NoMemory();
    }

    // Compute scores
    peachbase_batch_bm25_score(
        (const int*)tf_buf.buf,
        (const float*)idf_buf.buf,
        (const int*)dl_buf.buf,
        n_docs,
        n_terms,
        avg_doc_len,
        k1,
        b,
        results
    );

    // Convert to Python list
    PyObject* result_list = PyList_New(n_docs);
    for (size_t i = 0; i < n_docs; i++) {
        PyList_SET_ITEM(result_list, i, PyFloat_FromDouble((double)results[i]));
    }

    free(results);
    PyBuffer_Release(&tf_buf);
    PyBuffer_Release(&idf_buf);
    PyBuffer_Release(&dl_buf);

    return result_list;
}

static PyMethodDef Bm25Methods[] = {
    {"bm25_score", py_bm25_score, METH_VARARGS,
     "Compute BM25 score for a single document"},
    {"batch_bm25_score", py_batch_bm25_score, METH_VARARGS,
     "Batch compute BM25 scores for multiple documents"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bm25module = {
    PyModuleDef_HEAD_INIT,
    "_bm25",
    "Optimized BM25 scoring for PeachBase",
    -1,
    Bm25Methods
};

PyMODINIT_FUNC PyInit__bm25(void) {
    return PyModule_Create(&bm25module);
}
