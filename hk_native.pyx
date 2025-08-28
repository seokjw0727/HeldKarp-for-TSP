# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint64_t, int64_t, uint32_t
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as cnp

ctypedef cnp.int32_t INT32
ctypedef long long LL
ctypedef unsigned long long ULL

cdef inline LL INF64():
    return <LL>(1<<60)

# --------- Simple open-addressing hash table (linear probing) ---------
cdef struct HT:
    size_t cap
    size_t used
    ULL* keys
    LL*  vals

cdef inline size_t next_pow2(size_t x):
    if x <= 1:
        return 1
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    return x + 1

# A prime multiplier for 64-bit multiplicative hashing, related to the golden ratio.
cdef const ULL HASH_MULTIPLIER_64 = <ULL>11400714819323198485

cdef inline size_t hash64(ULL k, size_t cap_mask):
    return <size_t>((k * HASH_MULTIPLIER_64) & cap_mask)

cdef void ht_init(HT* ht, size_t init_cap):
    ht.cap = next_pow2(init_cap)
    if ht.cap < 16:
        ht.cap = 16
    ht.used = 0
    ht.keys = <ULL*> malloc(ht.cap * sizeof(ULL))
    ht.vals = <LL*>  malloc(ht.cap * sizeof(LL))
    if ht.keys == NULL or ht.vals == NULL:
        raise MemoryError("Failed to allocate HT arrays")
    memset(ht.keys, 0, ht.cap * sizeof(ULL))

cdef void ht_free(HT* ht):
    if ht.keys != NULL:
        free(ht.keys)
        ht.keys = NULL
    if ht.vals != NULL:
        free(ht.vals)
    ht.vals = NULL
    ht.cap = 0
    ht.used = 0

cdef inline void ht_rehash(HT* ht, size_t new_cap):
    cdef HT newht
    ht_init(&newht, new_cap)
    cdef size_t i, mask = newht.cap - 1
    cdef ULL k
    cdef size_t idx
    for i in range(ht.cap):
        k = ht.keys[i]
        if k != 0:
            idx = hash64(k, mask)
            while newht.keys[idx] != 0:
                idx = (idx + 1) & mask
            newht.keys[idx] = k
            newht.vals[idx] = ht.vals[i]
            newht.used += 1
    free(ht.keys)
    free(ht.vals)
    ht.cap = newht.cap
    ht.used = newht.used
    ht.keys = newht.keys
    ht.vals = newht.vals

cdef inline void ht_maybe_grow(HT* ht):
    if (ht.used + 1) * 4 >= ht.cap * 3:
        ht_rehash(ht, ht.cap << 1)

cdef inline bint ht_get(HT* ht, ULL key, LL* out_val):
    cdef size_t mask = ht.cap - 1
    cdef size_t idx = hash64(key, mask)
    cdef ULL k
    while True:
        k = ht.keys[idx]
        if k == 0:
            return False
        if k == key:
            out_val[0] = ht.vals[idx]
            return True
        idx = (idx + 1) & mask

cdef inline void ht_set_min(HT* ht, ULL key, LL val):
    ht_maybe_grow(ht)
    cdef size_t mask = ht.cap - 1
    cdef size_t idx = hash64(key, mask)
    while True:
        if ht.keys[idx] == 0:
            ht.keys[idx] = key
            ht.vals[idx] = val
            ht.used += 1
            return
        elif ht.keys[idx] == key:
            if val < ht.vals[idx]:
                ht.vals[idx] = val
            return
        idx = (idx + 1) & mask

# --------- C-level combinations of range(1..n-1) choose k ---------
cdef inline void comb_init(int k, int m, int* a):
    cdef int i
    for i in range(k):
        a[i] = i

cdef inline bint comb_next(int k, int m, int* a):
    cdef int i = k - 1
    while i >= 0 and a[i] == i + m - k:
        i -= 1
    if i < 0:
        return False
    a[i] += 1
    cdef int j
    for j in range(i+1, k):
        a[j] = a[j-1] + 1
    return True

# --------- Held-Karp optimized (native) ---------
cdef inline ULL pack_key(int mask, int idx):  # key = (mask << 6) | idx
    return ((<ULL>mask) << 6) | (<ULL>idx)

cpdef long long held_karp_optimized_native(object W_py):
    import numpy as np
    cdef cnp.ndarray[INT32, ndim=2] W = np.asarray(W_py, dtype=np.int32, order='C')
    cdef int n = W.shape[0]
    if n <= 1:
        return 0
    cdef int start = 0
    cdef LL INF = INF64()

    cdef HT dp_prev, dp_cur
    ht_init(&dp_prev, 1)
    ht_set_min(&dp_prev, pack_key(1<<start, start), 0)

    cdef int s, i, j, t, m
    cdef int k                # k = s-1
    cdef int others = n - 1   # vertices 1..n-1
    cdef int* a = NULL        # k-combination work array (size k)
    cdef int v, u
    cdef int mask, mask_wo, full_mask = (1<<n) - 1
    cdef LL best, prev, cand
    cdef ULL key_prev, key_cur
    cdef bint ok

    for s in range(2, n+1):
        k = s - 1
        comb_val = 1
        m = others
        for t in range(1, k+1):
            comb_val = (comb_val * (m - k + t)) // t
        expected = <size_t>(comb_val * k)
        ht_init(&dp_cur, (expected * 10) // 7 + 16)

        if a != NULL:
            free(a)
        a = <int*> malloc(k * sizeof(int))
        if a == NULL:
            ht_free(&dp_prev)
            ht_free(&dp_cur)
            raise MemoryError("Failed to allocate comb array")
        comb_init(k, m, a)

        while True:
            mask = 1<<start
            for t in range(k):
                v = a[t] + 1
                mask |= (1<<v)

            for t in range(k):
                j = a[t] + 1
                best = INF
                mask_wo = mask ^ (1<<j)

                key_prev = pack_key(mask_wo, start)
                ok = ht_get(&dp_prev, key_prev, &prev)
                if ok:
                    cand = prev + <LL>W[start, j]
                    if cand < best:
                        best = cand

                for u in range(k):
                    if u == t:
                        continue
                    i = a[u] + 1
                    key_prev = pack_key(mask_wo, i)
                    ok = ht_get(&dp_prev, key_prev, &prev)
                    if ok:
                        cand = prev + <LL>W[i, j]
                        if cand < best:
                            best = cand

                if best < INF:
                    key_cur = pack_key(mask, j)
                    ht_set_min(&dp_cur, key_cur, best)

            if not comb_next(k, m, a):
                break

        ht_free(&dp_prev)
        dp_prev = dp_cur
        memset(&dp_cur, 0, sizeof(HT))

    if a != NULL:
        free(a)

    cdef LL best_cost = INF
    for j in range(1, n):
        key_prev = pack_key(full_mask, j)
        if ht_get(&dp_prev, key_prev, &prev):
            cand = prev + <LL>W[j, start]
            if cand < best_cost:
                best_cost = cand

    ht_free(&dp_prev)
    return best_cost