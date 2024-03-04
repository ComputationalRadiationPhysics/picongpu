/* Copyright 2011 - 2023 Mutsuo Saito, Makoto Matsumoto, Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format off
#ifndef TINYMT32_H
#define TINYMT32_H
/**
 * @file tinymt32.h
 *
 * @brief Tiny Mersenne Twister only 127 bit internal state
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (University of Tokyo)
 *
 * Copyright (C) 2011 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */

#include "alpaka/core/BoostPredef.hpp"

#include <cstdint>
/* work-around for glibc < 2.18 according to bug
 * https://sourceware.org/bugzilla/show_bug.cgi?id=15366
 */
#ifndef UINT32_MAX
#   define UINT32_MAX ((uint32_t)-1u)
#endif
#ifndef UINT32_C
#   define UINT32_C(value) uint_least32_t(value)
#endif
#include <cinttypes>

#if BOOST_COMP_CLANG
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wold-style-cast"
#   pragma clang diagnostic ignored "-Wunused-function"
#endif
#if BOOST_COMP_GNUC
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
    #pragma warning(push)
    #pragma warning(disable: 4100)  // tinymt32.h(60): warning C4100: 'random': unreferenced formal parameter
#endif

#define TINYMT32_MEXP 127
#define TINYMT32_SH0 1
#define TINYMT32_SH1 10
#define TINYMT32_SH8 8
#define TINYMT32_MASK UINT32_C(0x7fffffff)
#define TINYMT32_MUL (1.0f / 16777216.0f)

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * tinymt32 internal state vector and parameters
 */
struct TINYMT32_T {
    uint32_t status[4];
    uint32_t mat1;
    uint32_t mat2;
    uint32_t tmat;
};

typedef struct TINYMT32_T tinymt32_t;

inline void tinymt32_init(tinymt32_t * random, uint32_t seed);
inline void tinymt32_init_by_array(tinymt32_t * random, uint32_t init_key[],
                            int key_length);

#if defined(__GNUC__)
/**
 * This function always returns 127
 * @param random not used
 * @return always 127
 */
inline static int tinymt32_get_mexp(
    tinymt32_t * random  __attribute__((unused))) {
    return TINYMT32_MEXP;
}
#else
inline static int tinymt32_get_mexp(tinymt32_t * random) {
    return TINYMT32_MEXP;
}
#endif

/**
 * This function changes internal state of tinymt32.
 * Users should not call this function directly.
 * @param random tinymt internal status
 */
inline static void tinymt32_next_state(tinymt32_t * random) {
    uint32_t x;
    uint32_t y;

    y = random->status[3];
    x = (random->status[0] & TINYMT32_MASK)
        ^ random->status[1]
        ^ random->status[2];
    x ^= (x << TINYMT32_SH0);
    y ^= (y >> TINYMT32_SH0) ^ x;
    random->status[0] = random->status[1];
    random->status[1] = random->status[2];
    random->status[2] = x ^ (y << TINYMT32_SH1);
    random->status[3] = y;
    int32_t const a = -((int32_t)(y & 1)) & (int32_t)random->mat1;
    int32_t const b = -((int32_t)(y & 1)) & (int32_t)random->mat2;
    random->status[1] ^= (uint32_t)a;
    random->status[2] ^= (uint32_t)b;
}

/**
 * This function outputs 32-bit unsigned integer from internal state.
 * Users should not call this function directly.
 * @param random tinymt internal status
 * @return 32-bit unsigned pseudorandom number
 */
inline static uint32_t tinymt32_temper(tinymt32_t * random) {
    uint32_t t0, t1;
    t0 = random->status[3];
#if defined(LINEARITY_CHECK)
    t1 = random->status[0]
        ^ (random->status[2] >> TINYMT32_SH8);
#else
    t1 = random->status[0]
        + (random->status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    if ((t1 & 1) != 0) {
        t0 ^= random->tmat;
    }
    return t0;
}

/**
 * This function outputs floating point number from internal state.
 * Users should not call this function directly.
 * @param random tinymt internal status
 * @return floating point number r (1.0 <= r < 2.0)
 */
inline static float tinymt32_temper_conv(tinymt32_t * random) {
    uint32_t t0, t1;
    union {
        uint32_t u;
        float f;
    } conv;

    t0 = random->status[3];
#if defined(LINEARITY_CHECK)
    t1 = random->status[0]
        ^ (random->status[2] >> TINYMT32_SH8);
#else
    t1 = random->status[0]
        + (random->status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    if ((t1 & 1) != 0) {
        conv.u  = ((t0 ^ random->tmat) >> 9) | UINT32_C(0x3f800000);
    } else {
        conv.u  = (t0 >> 9) | UINT32_C(0x3f800000);
    }
    return conv.f;
}

/**
 * This function outputs floating point number from internal state.
 * Users should not call this function directly.
 * @param random tinymt internal status
 * @return floating point number r (1.0 < r < 2.0)
 */
inline static float tinymt32_temper_conv_open(tinymt32_t * random) {
    uint32_t t0, t1;
    union {
        uint32_t u;
        float f;
    } conv;

    t0 = random->status[3];
#if defined(LINEARITY_CHECK)
    t1 = random->status[0]
        ^ (random->status[2] >> TINYMT32_SH8);
#else
    t1 = random->status[0]
        + (random->status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    if ((t1 & 1) != 0) {
        conv.u  = ((t0 ^ random->tmat) >> 9) | UINT32_C(0x3f800001);
    } else {
        conv.u  = (t0 >> 9) | UINT32_C(0x3f800001);
    }
    return conv.f;
}

/**
 * This function outputs 32-bit unsigned integer from internal state.
 * @param random tinymt internal status
 * @return 32-bit unsigned integer r (0 <= r < 2^32)
 */
inline static uint32_t tinymt32_generate_uint32(tinymt32_t * random) {
    tinymt32_next_state(random);
    return tinymt32_temper(random);
}

/**
 * This function outputs floating point number from internal state.
 * This function is implemented using multiplying by (1 / 2^24).
 * floating point multiplication is faster than using union trick in
 * my Intel CPU.
 * @param random tinymt internal status
 * @return floating point number r (0.0 <= r < 1.0)
 */
inline static float tinymt32_generate_float(tinymt32_t * random) {
    tinymt32_next_state(random);
    return (float)(tinymt32_temper(random) >> 8) * TINYMT32_MUL;
}

/**
 * This function outputs floating point number from internal state.
 * This function is implemented using union trick.
 * @param random tinymt internal status
 * @return floating point number r (1.0 <= r < 2.0)
 */
inline static float tinymt32_generate_float12(tinymt32_t * random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv(random);
}

/**
 * This function outputs floating point number from internal state.
 * This function is implemented using union trick.
 * @param random tinymt internal status
 * @return floating point number r (0.0 <= r < 1.0)
 */
inline static float tinymt32_generate_float01(tinymt32_t * random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv(random) - 1.0f;
}

/**
 * This function outputs floating point number from internal state.
 * This function may return 1.0 and never returns 0.0.
 * @param random tinymt internal status
 * @return floating point number r (0.0 < r <= 1.0)
 */
inline static float tinymt32_generate_floatOC(tinymt32_t * random) {
    tinymt32_next_state(random);
    return 1.0f - tinymt32_generate_float(random);
}

/**
 * This function outputs floating point number from internal state.
 * This function returns neither 0.0 nor 1.0.
 * @param random tinymt internal status
 * @return floating point number r (0.0 < r < 1.0)
 */
inline static float tinymt32_generate_floatOO(tinymt32_t * random) {
    tinymt32_next_state(random);
    return tinymt32_temper_conv_open(random) - 1.0f;
}

/**
 * This function outputs double precision floating point number from
 * internal state. The returned value has 32-bit precision.
 * In other words, this function makes one double precision floating point
 * number from one 32-bit unsigned integer.
 * @param random tinymt internal status
 * @return floating point number r (0.0 <= r < 1.0)
 */
inline static double tinymt32_generate_32double(tinymt32_t * random) {
    tinymt32_next_state(random);
    return tinymt32_temper(random) * (1.0 / 4294967296.0);
}

#if defined(__cplusplus)
}
#endif

#define MIN_LOOP 8
#define PRE_LOOP 8

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func1(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1664525);
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func2(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1566083941);
}

/**
 * This function certificate the period of 2^127-1.
 * @param random tinymt state vector.
 */
static void period_certification(tinymt32_t * random) {
    if ((random->status[0] & TINYMT32_MASK) == 0 &&
        random->status[1] == 0 &&
        random->status[2] == 0 &&
        random->status[3] == 0) {
        random->status[0] = 'T';
        random->status[1] = 'I';
        random->status[2] = 'N';
        random->status[3] = 'Y';
    }
}

/**
 * This function initializes the internal state array with a 32-bit
 * unsigned integer seed.
 * @param random tinymt state vector.
 * @param seed a 32-bit unsigned integer used as a seed.
 */
void tinymt32_init(tinymt32_t * random, uint32_t seed) {
    random->status[0] = seed;
    random->status[1] = random->mat1;
    random->status[2] = random->mat2;
    random->status[3] = random->tmat;
    for (unsigned int i = 1; i < MIN_LOOP; i++) {
        random->status[i & 3] ^= i + UINT32_C(1812433253)
            * (random->status[(i - 1) & 3]
               ^ (random->status[(i - 1) & 3] >> 30));
    }
    period_certification(random);
    for (unsigned int i = 0; i < PRE_LOOP; i++) {
        tinymt32_next_state(random);
    }
}

/**
 * This function initializes the internal state array,
 * with an array of 32-bit unsigned integers used as seeds
 * @param random tinymt state vector.
 * @param init_key the array of 32-bit integers, used as a seed.
 * @param key_length the length of init_key.
 */
void tinymt32_init_by_array(tinymt32_t * random, uint32_t init_key[],
                            int key_length) {
    const unsigned int lag = 1;
    const unsigned int mid = 1;
    const unsigned int size = 4;
    unsigned int i, j;
    unsigned int count;
    uint32_t r;
    uint32_t * st = &random->status[0];

    st[0] = 0;
    st[1] = random->mat1;
    st[2] = random->mat2;
    st[3] = random->tmat;
    if (key_length + 1 > MIN_LOOP) {
        count = (unsigned int)key_length + 1;
    } else {
        count = MIN_LOOP;
    }
    r = ini_func1(st[0] ^ st[mid % size]
                  ^ st[(size - 1) % size]);
    st[mid % size] += r;
    r += (unsigned int)key_length;
    st[(mid + lag) % size] += r;
    st[0] = r;
    count--;
    for (i = 1, j = 0; (j < count) && (j < (unsigned int)key_length); j++) {
        r = ini_func1(st[i % size]
                      ^ st[(i + mid) % size]
                      ^ st[(i + size - 1) % size]);
        st[(i + mid) % size] += r;
        r += init_key[j] + i;
        st[(i + mid + lag) % size] += r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    for (; j < count; j++) {
        r = ini_func1(st[i % size]
                      ^ st[(i + mid) % size]
                      ^ st[(i + size - 1) % size]);
        st[(i + mid) % size] += r;
        r += i;
        st[(i + mid + lag) % size] += r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
        r = ini_func2(st[i % size]
                      + st[(i + mid) % size]
                      + st[(i + size - 1) % size]);
        st[(i + mid) % size] ^= r;
        r -= i;
        st[(i + mid + lag) % size] ^= r;
        st[i % size] = r;
        i = (i + 1) % size;
    }
    period_certification(random);
    for (i = 0; i < PRE_LOOP; i++) {
        tinymt32_next_state(random);
    }
}

#undef MIN_LOOP
#undef PRE_LOOP

#if BOOST_COMP_CLANG
#   pragma clang diagnostic pop
#endif
#if BOOST_COMP_GNUC
#   pragma GCC diagnostic pop
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#   pragma warning(pop)
#endif

#endif
