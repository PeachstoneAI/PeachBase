/**
 * peachbase_simd.c
 *
 * SIMD-optimized vector similarity operations for PeachBase.
 * Implements AVX2, AVX-512 (future), and fallback versions.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "peachbase_simd.h"
#include <math.h>
#include <string.h>

// Include SIMD intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define PEACHBASE_X86
#endif

// Global variable to store detected CPU features
static peachbase_cpu_features_t g_cpu_features = PEACHBASE_CPU_FALLBACK;
static int g_cpu_detected = 0;

/**
 * Detect CPU features at runtime using CPUID.
 */
peachbase_cpu_features_t peachbase_detect_cpu_features(void) {
    if (g_cpu_detected) {
        return g_cpu_features;
    }

#ifdef PEACHBASE_X86
    // Use GCC/Clang built-in CPU feature detection
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();

        // Check for AVX-512 first (implies AVX2)
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")) {
            g_cpu_features = PEACHBASE_CPU_AVX512;
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            g_cpu_features = PEACHBASE_CPU_AVX2;
        }
    #endif
#endif

    g_cpu_detected = 1;
    return g_cpu_features;
}

/* ========================================================================
 * Fallback implementations (portable, no SIMD)
 * ======================================================================== */

static float cosine_similarity_fallback(const float* vec1, const float* vec2, size_t dim) {
    float dot = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < dim; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    float denom = sqrtf(norm1) * sqrtf(norm2);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

static float l2_distance_fallback(const float* vec1, const float* vec2, size_t dim) {
    float sum = 0.0f;

    for (size_t i = 0; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }

    return sqrtf(sum);
}

static float dot_product_fallback(const float* vec1, const float* vec2, size_t dim) {
    float dot = 0.0f;

    for (size_t i = 0; i < dim; i++) {
        dot += vec1[i] * vec2[i];
    }

    return dot;
}

/* ========================================================================
 * Helper: Fast horizontal sum for AVX2
 * ======================================================================== */

#if defined(PEACHBASE_X86) && defined(__AVX2__)

// Fast horizontal sum using shuffle intrinsics
static inline float hsum_avx2(__m256 v) {
    // Add high and low 128-bit lanes
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);

    // Horizontal add within 128-bit lane
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

/* ========================================================================
 * AVX2 implementations
 * ======================================================================== */

static float cosine_similarity_avx2(const float* vec1, const float* vec2, size_t dim) {
    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_norm1 = _mm256_setzero_ps();
    __m256 sum_norm2 = _mm256_setzero_ps();

    size_t i = 0;
    // Process 8 floats at a time
    for (; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);

        sum_dot = _mm256_fmadd_ps(v1, v2, sum_dot);
        sum_norm1 = _mm256_fmadd_ps(v1, v1, sum_norm1);
        sum_norm2 = _mm256_fmadd_ps(v2, v2, sum_norm2);
    }

    // Fast horizontal sum using intrinsics
    float dot = hsum_avx2(sum_dot);
    float norm1 = hsum_avx2(sum_norm1);
    float norm2 = hsum_avx2(sum_norm2);

    // Handle remaining elements
    for (; i < dim; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    float denom = sqrtf(norm1) * sqrtf(norm2);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

static float l2_distance_avx2(const float* vec1, const float* vec2, size_t dim) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);
        __m256 diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Fast horizontal sum
    float result = hsum_avx2(sum);

    // Handle remaining elements
    for (; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        result += diff * diff;
    }

    return sqrtf(result);
}

static float dot_product_avx2(const float* vec1, const float* vec2, size_t dim) {
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);
        sum = _mm256_fmadd_ps(v1, v2, sum);
    }

    // Fast horizontal sum
    float result = hsum_avx2(sum);

    // Handle remaining elements
    for (; i < dim; i++) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

#endif /* AVX2 */

/* ========================================================================
 * AVX-512 implementations
 * ======================================================================== */

#if defined(PEACHBASE_X86) && defined(__AVX512F__)

// Fast horizontal sum for AVX-512
static inline float hsum_avx512(__m512 v) {
    // Reduce 512-bit to 256-bit
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);

    // Use AVX2 horizontal sum for the rest
    return hsum_avx2(sum256);
}

static float cosine_similarity_avx512(const float* vec1, const float* vec2, size_t dim) {
    __m512 sum_dot = _mm512_setzero_ps();
    __m512 sum_norm1 = _mm512_setzero_ps();
    __m512 sum_norm2 = _mm512_setzero_ps();

    size_t i = 0;
    // Process 16 floats at a time
    for (; i + 16 <= dim; i += 16) {
        __m512 v1 = _mm512_loadu_ps(&vec1[i]);
        __m512 v2 = _mm512_loadu_ps(&vec2[i]);

        sum_dot = _mm512_fmadd_ps(v1, v2, sum_dot);
        sum_norm1 = _mm512_fmadd_ps(v1, v1, sum_norm1);
        sum_norm2 = _mm512_fmadd_ps(v2, v2, sum_norm2);
    }

    // Fast horizontal sum
    float dot = hsum_avx512(sum_dot);
    float norm1 = hsum_avx512(sum_norm1);
    float norm2 = hsum_avx512(sum_norm2);

    // Handle remaining elements
    for (; i < dim; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    float denom = sqrtf(norm1) * sqrtf(norm2);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

static float l2_distance_avx512(const float* vec1, const float* vec2, size_t dim) {
    __m512 sum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 v1 = _mm512_loadu_ps(&vec1[i]);
        __m512 v2 = _mm512_loadu_ps(&vec2[i]);
        __m512 diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    float result = hsum_avx512(sum);

    // Handle remaining elements
    for (; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        result += diff * diff;
    }

    return sqrtf(result);
}

static float dot_product_avx512(const float* vec1, const float* vec2, size_t dim) {
    __m512 sum = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 v1 = _mm512_loadu_ps(&vec1[i]);
        __m512 v2 = _mm512_loadu_ps(&vec2[i]);
        sum = _mm512_fmadd_ps(v1, v2, sum);
    }

    float result = hsum_avx512(sum);

    // Handle remaining elements
    for (; i < dim; i++) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

#endif /* AVX512 */

/* ========================================================================
 * Public API implementations
 * ======================================================================== */

float peachbase_cosine_similarity_simd(const float* vec1, const float* vec2, size_t dim) {
    if (!g_cpu_detected) {
        peachbase_detect_cpu_features();
    }

#if defined(PEACHBASE_X86) && defined(__AVX512F__)
    if (g_cpu_features >= PEACHBASE_CPU_AVX512) {
        return cosine_similarity_avx512(vec1, vec2, dim);
    }
#endif

#if defined(PEACHBASE_X86) && defined(__AVX2__)
    if (g_cpu_features >= PEACHBASE_CPU_AVX2) {
        return cosine_similarity_avx2(vec1, vec2, dim);
    }
#endif

    return cosine_similarity_fallback(vec1, vec2, dim);
}

float peachbase_l2_distance_simd(const float* vec1, const float* vec2, size_t dim) {
    if (!g_cpu_detected) {
        peachbase_detect_cpu_features();
    }

#if defined(PEACHBASE_X86) && defined(__AVX512F__)
    if (g_cpu_features >= PEACHBASE_CPU_AVX512) {
        return l2_distance_avx512(vec1, vec2, dim);
    }
#endif

#if defined(PEACHBASE_X86) && defined(__AVX2__)
    if (g_cpu_features >= PEACHBASE_CPU_AVX2) {
        return l2_distance_avx2(vec1, vec2, dim);
    }
#endif

    return l2_distance_fallback(vec1, vec2, dim);
}

float peachbase_dot_product_simd(const float* vec1, const float* vec2, size_t dim) {
    if (!g_cpu_detected) {
        peachbase_detect_cpu_features();
    }

#if defined(PEACHBASE_X86) && defined(__AVX512F__)
    if (g_cpu_features >= PEACHBASE_CPU_AVX512) {
        return dot_product_avx512(vec1, vec2, dim);
    }
#endif

#if defined(PEACHBASE_X86) && defined(__AVX2__)
    if (g_cpu_features >= PEACHBASE_CPU_AVX2) {
        return dot_product_avx2(vec1, vec2, dim);
    }
#endif

    return dot_product_fallback(vec1, vec2, dim);
}

void peachbase_batch_cosine_similarity(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
) {
    // Only use OpenMP for larger batches (overhead not worth it for small batches)
    // Note: OpenMP 2.0 (MSVC) requires signed loop index
    int i;
    int n = (int)n_vectors;
#ifdef _OPENMP
    if (n_vectors >= 1000) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (i = 0; i < n; i++) {
            results[i] = peachbase_cosine_similarity_simd(query, &vectors[i * dim], dim);
        }
    } else {
        for (i = 0; i < n; i++) {
            results[i] = peachbase_cosine_similarity_simd(query, &vectors[i * dim], dim);
        }
    }
#else
    // OpenMP not available, use sequential
    for (i = 0; i < n; i++) {
        results[i] = peachbase_cosine_similarity_simd(query, &vectors[i * dim], dim);
    }
#endif
}

void peachbase_batch_l2_distance(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
) {
    // Only use OpenMP for larger batches (overhead not worth it for small batches)
    // Note: OpenMP 2.0 (MSVC) requires signed loop index
    int i;
    int n = (int)n_vectors;
#ifdef _OPENMP
    if (n_vectors >= 1000) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (i = 0; i < n; i++) {
            results[i] = peachbase_l2_distance_simd(query, &vectors[i * dim], dim);
        }
    } else {
        for (i = 0; i < n; i++) {
            results[i] = peachbase_l2_distance_simd(query, &vectors[i * dim], dim);
        }
    }
#else
    // OpenMP not available, use sequential
    for (i = 0; i < n; i++) {
        results[i] = peachbase_l2_distance_simd(query, &vectors[i * dim], dim);
    }
#endif
}

void peachbase_batch_dot_product(
    const float* query,
    const float* vectors,
    size_t n_vectors,
    size_t dim,
    float* results
) {
    // Only use OpenMP for larger batches (overhead not worth it for small batches)
    // Note: OpenMP 2.0 (MSVC) requires signed loop index
    int i;
    int n = (int)n_vectors;
#ifdef _OPENMP
    if (n_vectors >= 1000) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (i = 0; i < n; i++) {
            results[i] = peachbase_dot_product_simd(query, &vectors[i * dim], dim);
        }
    } else {
        for (i = 0; i < n; i++) {
            results[i] = peachbase_dot_product_simd(query, &vectors[i * dim], dim);
        }
    }
#else
    // OpenMP not available, use sequential
    for (i = 0; i < n; i++) {
        results[i] = peachbase_dot_product_simd(query, &vectors[i * dim], dim);
    }
#endif
}

/* ========================================================================
 * Python bindings
 * ======================================================================== */

static PyObject* py_detect_cpu_features(PyObject* self, PyObject* args) {
    peachbase_cpu_features_t features = peachbase_detect_cpu_features();
    return PyLong_FromLong((long)features);
}

static PyObject* py_get_openmp_info(PyObject* self, PyObject* args) {
    PyObject* info_dict = PyDict_New();

#ifdef _OPENMP
    PyDict_SetItemString(info_dict, "compiled_with_openmp", Py_True);

    // Get OpenMP version
    int omp_version = _OPENMP;
    PyDict_SetItemString(info_dict, "openmp_version", PyLong_FromLong(omp_version));

    // Get number of threads (must be called from parallel region for accuracy)
    // For now, just report max threads
    #include <omp.h>
    int max_threads = omp_get_max_threads();
    PyDict_SetItemString(info_dict, "max_threads", PyLong_FromLong(max_threads));

    // Get number of processors
    int num_procs = omp_get_num_procs();
    PyDict_SetItemString(info_dict, "num_procs", PyLong_FromLong(num_procs));
#else
    PyDict_SetItemString(info_dict, "compiled_with_openmp", Py_False);
    PyDict_SetItemString(info_dict, "openmp_version", Py_None);
    PyDict_SetItemString(info_dict, "max_threads", PyLong_FromLong(1));
    PyDict_SetItemString(info_dict, "num_procs", PyLong_FromLong(1));
#endif

    return info_dict;
}

static PyObject* py_cosine_similarity(PyObject* self, PyObject* args) {
    PyObject *vec1_obj, *vec2_obj;

    if (!PyArg_ParseTuple(args, "OO", &vec1_obj, &vec2_obj)) {
        return NULL;
    }

    Py_buffer vec1_buf, vec2_buf;
    if (PyObject_GetBuffer(vec1_obj, &vec1_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(vec2_obj, &vec2_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&vec1_buf);
        return NULL;
    }

    size_t dim1 = vec1_buf.len / sizeof(float);
    size_t dim2 = vec2_buf.len / sizeof(float);

    if (dim1 != dim2) {
        PyBuffer_Release(&vec1_buf);
        PyBuffer_Release(&vec2_buf);
        PyErr_SetString(PyExc_ValueError, "Vector dimensions must match");
        return NULL;
    }

    float result = peachbase_cosine_similarity_simd(
        (const float*)vec1_buf.buf,
        (const float*)vec2_buf.buf,
        dim1
    );

    PyBuffer_Release(&vec1_buf);
    PyBuffer_Release(&vec2_buf);

    return PyFloat_FromDouble((double)result);
}

static PyObject* py_l2_distance(PyObject* self, PyObject* args) {
    PyObject *vec1_obj, *vec2_obj;

    if (!PyArg_ParseTuple(args, "OO", &vec1_obj, &vec2_obj)) {
        return NULL;
    }

    Py_buffer vec1_buf, vec2_buf;
    if (PyObject_GetBuffer(vec1_obj, &vec1_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(vec2_obj, &vec2_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&vec1_buf);
        return NULL;
    }

    size_t dim1 = vec1_buf.len / sizeof(float);
    size_t dim2 = vec2_buf.len / sizeof(float);

    if (dim1 != dim2) {
        PyBuffer_Release(&vec1_buf);
        PyBuffer_Release(&vec2_buf);
        PyErr_SetString(PyExc_ValueError, "Vector dimensions must match");
        return NULL;
    }

    float result = peachbase_l2_distance_simd(
        (const float*)vec1_buf.buf,
        (const float*)vec2_buf.buf,
        dim1
    );

    PyBuffer_Release(&vec1_buf);
    PyBuffer_Release(&vec2_buf);

    return PyFloat_FromDouble((double)result);
}

static PyObject* py_dot_product(PyObject* self, PyObject* args) {
    PyObject *vec1_obj, *vec2_obj;

    if (!PyArg_ParseTuple(args, "OO", &vec1_obj, &vec2_obj)) {
        return NULL;
    }

    Py_buffer vec1_buf, vec2_buf;
    if (PyObject_GetBuffer(vec1_obj, &vec1_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(vec2_obj, &vec2_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&vec1_buf);
        return NULL;
    }

    size_t dim1 = vec1_buf.len / sizeof(float);
    size_t dim2 = vec2_buf.len / sizeof(float);

    if (dim1 != dim2) {
        PyBuffer_Release(&vec1_buf);
        PyBuffer_Release(&vec2_buf);
        PyErr_SetString(PyExc_ValueError, "Vector dimensions must match");
        return NULL;
    }

    float result = peachbase_dot_product_simd(
        (const float*)vec1_buf.buf,
        (const float*)vec2_buf.buf,
        dim1
    );

    PyBuffer_Release(&vec1_buf);
    PyBuffer_Release(&vec2_buf);

    return PyFloat_FromDouble((double)result);
}

static PyObject* py_batch_cosine_similarity(PyObject* self, PyObject* args) {
    PyObject *query_obj, *vectors_obj;

    if (!PyArg_ParseTuple(args, "OO", &query_obj, &vectors_obj)) {
        return NULL;
    }

    Py_buffer query_buf, vectors_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(vectors_obj, &vectors_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&query_buf);
        return NULL;
    }

    size_t dim = query_buf.len / sizeof(float);
    size_t n_vectors = vectors_buf.len / sizeof(float) / dim;

    float* results = (float*)malloc(n_vectors * sizeof(float));
    if (!results) {
        PyBuffer_Release(&query_buf);
        PyBuffer_Release(&vectors_buf);
        return PyErr_NoMemory();
    }

    // Release GIL during computation for true parallelism
    Py_BEGIN_ALLOW_THREADS
    peachbase_batch_cosine_similarity(
        (const float*)query_buf.buf,
        (const float*)vectors_buf.buf,
        n_vectors,
        dim,
        results
    );
    Py_END_ALLOW_THREADS

    PyObject* result_list = PyList_New(n_vectors);
    for (size_t i = 0; i < n_vectors; i++) {
        PyList_SET_ITEM(result_list, i, PyFloat_FromDouble((double)results[i]));
    }

    free(results);
    PyBuffer_Release(&query_buf);
    PyBuffer_Release(&vectors_buf);

    return result_list;
}

static PyObject* py_batch_l2_distance(PyObject* self, PyObject* args) {
    PyObject *query_obj, *vectors_obj;

    if (!PyArg_ParseTuple(args, "OO", &query_obj, &vectors_obj)) {
        return NULL;
    }

    Py_buffer query_buf, vectors_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(vectors_obj, &vectors_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&query_buf);
        return NULL;
    }

    size_t dim = query_buf.len / sizeof(float);
    size_t n_vectors = vectors_buf.len / sizeof(float) / dim;

    float* results = (float*)malloc(n_vectors * sizeof(float));
    if (!results) {
        PyBuffer_Release(&query_buf);
        PyBuffer_Release(&vectors_buf);
        return PyErr_NoMemory();
    }

    // Release GIL during computation for true parallelism
    Py_BEGIN_ALLOW_THREADS
    peachbase_batch_l2_distance(
        (const float*)query_buf.buf,
        (const float*)vectors_buf.buf,
        n_vectors,
        dim,
        results
    );
    Py_END_ALLOW_THREADS

    PyObject* result_list = PyList_New(n_vectors);
    for (size_t i = 0; i < n_vectors; i++) {
        PyList_SET_ITEM(result_list, i, PyFloat_FromDouble((double)results[i]));
    }

    free(results);
    PyBuffer_Release(&query_buf);
    PyBuffer_Release(&vectors_buf);

    return result_list;
}

static PyObject* py_batch_dot_product(PyObject* self, PyObject* args) {
    PyObject *query_obj, *vectors_obj;

    if (!PyArg_ParseTuple(args, "OO", &query_obj, &vectors_obj)) {
        return NULL;
    }

    Py_buffer query_buf, vectors_buf;
    if (PyObject_GetBuffer(query_obj, &query_buf, PyBUF_SIMPLE) != 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(vectors_obj, &vectors_buf, PyBUF_SIMPLE) != 0) {
        PyBuffer_Release(&query_buf);
        return NULL;
    }

    size_t dim = query_buf.len / sizeof(float);
    size_t n_vectors = vectors_buf.len / sizeof(float) / dim;

    float* results = (float*)malloc(n_vectors * sizeof(float));
    if (!results) {
        PyBuffer_Release(&query_buf);
        PyBuffer_Release(&vectors_buf);
        return PyErr_NoMemory();
    }

    // Release GIL during computation for true parallelism
    Py_BEGIN_ALLOW_THREADS
    peachbase_batch_dot_product(
        (const float*)query_buf.buf,
        (const float*)vectors_buf.buf,
        n_vectors,
        dim,
        results
    );
    Py_END_ALLOW_THREADS

    PyObject* result_list = PyList_New(n_vectors);
    for (size_t i = 0; i < n_vectors; i++) {
        PyList_SET_ITEM(result_list, i, PyFloat_FromDouble((double)results[i]));
    }

    free(results);
    PyBuffer_Release(&query_buf);
    PyBuffer_Release(&vectors_buf);

    return result_list;
}

static PyMethodDef SimdMethods[] = {
    {"detect_cpu_features", py_detect_cpu_features, METH_NOARGS,
     "Detect available CPU SIMD features"},
    {"get_openmp_info", py_get_openmp_info, METH_NOARGS,
     "Get OpenMP compilation and runtime information"},
    {"cosine_similarity", py_cosine_similarity, METH_VARARGS,
     "Compute cosine similarity between two vectors"},
    {"l2_distance", py_l2_distance, METH_VARARGS,
     "Compute L2 distance between two vectors"},
    {"dot_product", py_dot_product, METH_VARARGS,
     "Compute dot product between two vectors"},
    {"batch_cosine_similarity", py_batch_cosine_similarity, METH_VARARGS,
     "Batch compute cosine similarities"},
    {"batch_l2_distance", py_batch_l2_distance, METH_VARARGS,
     "Batch compute L2 distances"},
    {"batch_dot_product", py_batch_dot_product, METH_VARARGS,
     "Batch compute dot products"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef simdmodule = {
    PyModuleDef_HEAD_INIT,
    "_simd",
    "SIMD-optimized vector operations for PeachBase",
    -1,
    SimdMethods
};

PyMODINIT_FUNC PyInit__simd(void) {
    return PyModule_Create(&simdmodule);
}
