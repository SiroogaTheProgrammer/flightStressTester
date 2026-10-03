#pragma once

#include <cstdint>
#include <cstring>

#ifdef YOLO_HOST_TEST
#include <chrono>

inline uint32_t yolo_now_us() {
    using namespace std::chrono;
    return static_cast<uint32_t>(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
}
#define YOLO_HOT
#else
#include "pico/stdlib.h"

inline uint32_t yolo_now_us() { return time_us_32(); }
#ifdef YOLO_KERNELS_IN_RAM
#define YOLO_HOT __attribute__((section(".time_critical.yolo")))
#else
#define YOLO_HOT
#endif
#endif

#if defined(__ARM_FEATURE_DSP) && !defined(YOLO_HOST_TEST)
#include <arm_acle.h>

static inline uint32_t sxtb16(uint32_t x) { return __sxtb16(x); }
static inline uint32_t uxtb16(uint32_t x) { return __uxtb16(x); }
static inline uint32_t sxtb16_ror8(uint32_t x) {
    uint32_t r;
    __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(x));
    return r;
}
static inline uint32_t uxtb16_ror8(uint32_t x) {
    uint32_t r;
    __asm__("uxtb16 %0, %1, ror #8" : "=r"(r) : "r"(x));
    return r;
}
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t c) { return __smlad(a, b, c); }
static inline int32_t smlabb(uint32_t a, uint32_t b, int32_t c) { return __smlabb(a, b, c); }
static inline int32_t smlatt(uint32_t a, uint32_t b, int32_t c) { return __smlatt(a, b, c); }
static inline int32_t usat8(int32_t x) { return __usat(x, 8); }
static inline int32_t ssat8(int32_t x) { return __ssat(x, 8); }
static inline uint32_t uqadd8(uint32_t a, uint32_t b) { return __uqadd8(a, b); }
static inline uint32_t uqsub8(uint32_t a, uint32_t b) { return __uqsub8(a, b); }
#else
static inline uint32_t sxtb16(uint32_t x) {
    const uint32_t lo = static_cast<uint16_t>(static_cast<int16_t>(static_cast<int8_t>(x & 0xFF)));
    const uint32_t hi = static_cast<uint16_t>(static_cast<int16_t>(static_cast<int8_t>((x >> 16) & 0xFF)));
    return lo | (hi << 16);
}
static inline uint32_t uxtb16(uint32_t x) { return x & 0x00FF00FFu; }
static inline uint32_t sxtb16_ror8(uint32_t x) { return sxtb16((x >> 8) | (x << 24)); }
static inline uint32_t uxtb16_ror8(uint32_t x) { return uxtb16((x >> 8) | (x << 24)); }
static inline int32_t half_lo(uint32_t x) { return static_cast<int16_t>(x & 0xFFFF); }
static inline int32_t half_hi(uint32_t x) { return static_cast<int16_t>(x >> 16); }
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t c) {
    return c + half_lo(a) * half_lo(b) + half_hi(a) * half_hi(b);
}
static inline int32_t smlabb(uint32_t a, uint32_t b, int32_t c) { return c + half_lo(a) * half_lo(b); }
static inline int32_t smlatt(uint32_t a, uint32_t b, int32_t c) { return c + half_hi(a) * half_hi(b); }
static inline int32_t usat8(int32_t x) { return x < 0 ? 0 : (x > 255 ? 255 : x); }
static inline int32_t ssat8(int32_t x) { return x < -128 ? -128 : (x > 127 ? 127 : x); }
static inline uint32_t uqadd8(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 32; i += 8) {
        const uint32_t s = ((a >> i) & 0xFF) + ((b >> i) & 0xFF);
        r |= (s > 255 ? 255u : s) << i;
    }
    return r;
}
static inline uint32_t uqsub8(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 32; i += 8) {
        const uint32_t x = (a >> i) & 0xFF, y = (b >> i) & 0xFF;
        r |= (x > y ? x - y : 0u) << i;
    }
    return r;
}
#endif
