#include "yolo_engine.h"

#include <cstring>

#include "yolo_model_data.h"
#include "yolo_portable.h"

#define YOLO_NOINLINE __attribute__((noinline))

namespace yolo {
namespace {

constexpr int kConv1Channels = 24;
constexpr int kConv1Taps = 28;
constexpr uint32_t kHashPrime = 1000003u;

alignas(16) uint32_t g_weights[(kMaxWeightBytes + 15) / 4 + 4];
alignas(16) uint8_t g_ring[3][(kInputSize / 2) * kConv1Channels];
alignas(16) uint8_t g_im2col[(kInputSize / 2) * kConv1Taps];

inline uint32_t ld32(const void* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline void st32(void* p, uint32_t v) { std::memcpy(p, &v, sizeof(v)); }

template <bool S>
inline uint32_t ext_even(uint32_t x) { return S ? sxtb16(x) : uxtb16(x); }
template <bool S>
inline uint32_t ext_odd(uint32_t x) { return S ? sxtb16_ror8(x) : uxtb16_ror8(x); }
template <bool S>
inline int32_t sat8(int32_t x) { return S ? ssat8(x) : usat8(x); }

// Valid because every conv/dw layer has shift >= 33, so the rounding term only touches the high word.
inline int32_t requant(int32_t acc, int32_t mult, int32_t rnd, int shift) {
    return (static_cast<int32_t>((static_cast<int64_t>(acc) * mult) >> 32) + rnd) >> shift;
}

inline int32_t requant_generic(int32_t x, int32_t mult, int shift, int32_t lo, int32_t hi) {
    const int64_t v = static_cast<int64_t>(x) * mult + (static_cast<int64_t>(1) << (shift - 1));
    const int32_t r = static_cast<int32_t>(v >> shift);
    return r < lo ? lo : (r > hi ? hi : r);
}

template <bool OUT_S>
inline uint32_t pack4(int32_t a, int32_t b, int32_t c, int32_t d) {
    return (static_cast<uint32_t>(sat8<OUT_S>(a)) & 0xFF) | ((static_cast<uint32_t>(sat8<OUT_S>(b)) & 0xFF) << 8) |
           ((static_cast<uint32_t>(sat8<OUT_S>(c)) & 0xFF) << 16) | ((static_cast<uint32_t>(sat8<OUT_S>(d)) & 0xFF) << 24);
}

// Weights: per block of 4 output rows and per group of 4 inputs, 8 words (even/odd lane pairs of each row).
template <bool IN_S, bool OUT_S>
YOLO_HOT YOLO_NOINLINE void pointwise(int npix, int cin, int cout, const uint8_t* in, int in_stride, uint8_t* out,
                                      int out_stride, const uint32_t* w, const int32_t* bias, const int32_t* mult,
                                      int shift) {
    const int rs = shift - 32;
    const int32_t rnd = 1 << (rs - 1);
    const int groups = cin >> 2;
    const int blocks = cout >> 2;
    const uint32_t* tail = w + blocks * groups * 8;

    for (int p = 0; p < npix; ++p) {
        const uint8_t* row = in + p * in_stride;
        uint8_t* o = out + p * out_stride;
        const uint32_t* wp = w;
        for (int b = 0; b < blocks; ++b) {
            const int c = b << 2;
            int32_t s0 = bias[c], s1 = bias[c + 1], s2 = bias[c + 2], s3 = bias[c + 3];
            const uint8_t* a = row;
            for (int g = 0; g < groups; ++g) {
                const uint32_t x = ld32(a);
                a += 4;
                const uint32_t xe = ext_even<IN_S>(x), xo = ext_odd<IN_S>(x);
                s0 = smlad(xe, wp[0], s0);
                s0 = smlad(xo, wp[1], s0);
                s1 = smlad(xe, wp[2], s1);
                s1 = smlad(xo, wp[3], s1);
                s2 = smlad(xe, wp[4], s2);
                s2 = smlad(xo, wp[5], s2);
                s3 = smlad(xe, wp[6], s3);
                s3 = smlad(xo, wp[7], s3);
                wp += 8;
            }
            st32(o + c, pack4<OUT_S>(requant(s0, mult[c], rnd, rs), requant(s1, mult[c + 1], rnd, rs),
                                     requant(s2, mult[c + 2], rnd, rs), requant(s3, mult[c + 3], rnd, rs)));
        }
        const uint32_t* tw = tail;
        for (int r = blocks << 2; r < cout; ++r) {
            int32_t s = bias[r];
            const uint8_t* a = row;
            for (int g = 0; g < groups; ++g) {
                const uint32_t x = ld32(a);
                a += 4;
                s = smlad(ext_even<IN_S>(x), tw[0], s);
                s = smlad(ext_odd<IN_S>(x), tw[1], s);
                tw += 2;
            }
            o[r] = static_cast<uint8_t>(sat8<OUT_S>(requant(s, mult[r], rnd, rs)));
        }
    }
}

template <bool IN_S, bool OUT_S, int K>
inline void dw_groups(const uint8_t* ip0, int row_bytes, int in_stride, const uint32_t* w, int groups,
                      const int32_t* bias, const int32_t* mult, uint8_t* o, int ky0, int ky1, int kx0, int kx1,
                      int rs, int32_t rnd) {
    const int tap_words = groups * 2;
    for (int g = 0; g < groups; ++g) {
        const int c = g << 2;
        int32_t a0 = bias[c], a1 = bias[c + 1], a2 = bias[c + 2], a3 = bias[c + 3];
        const uint8_t* rowp = ip0 + c;
        const uint32_t* wrow = w + (ky0 * K + kx0) * tap_words + g * 2;
        for (int ky = ky0; ky < ky1; ++ky) {
            const uint8_t* ip = rowp;
            const uint32_t* wp = wrow;
            for (int kx = kx0; kx < kx1; ++kx) {
                const uint32_t x = ld32(ip);
                const uint32_t xe = ext_even<IN_S>(x), xo = ext_odd<IN_S>(x);
                const uint32_t we = wp[0], wo = wp[1];
                a0 = smlabb(xe, we, a0);
                a2 = smlatt(xe, we, a2);
                a1 = smlabb(xo, wo, a1);
                a3 = smlatt(xo, wo, a3);
                ip += in_stride;
                wp += tap_words;
            }
            rowp += row_bytes;
            wrow += K * tap_words;
        }
        st32(o + c, pack4<OUT_S>(requant(a0, mult[c], rnd, rs), requant(a1, mult[c + 1], rnd, rs),
                                 requant(a2, mult[c + 2], rnd, rs), requant(a3, mult[c + 3], rnd, rs)));
    }
}

template <bool IN_S, bool OUT_S, int K>
YOLO_HOT YOLO_NOINLINE void depthwise(const Layer& L, const uint8_t* in, uint8_t* out, const uint32_t* w,
                                      const int32_t* bias, const int32_t* mult) {
    const int pad = K >> 1, stride = L.stride, groups = L.cin >> 2;
    const int height = L.h, width = L.w, out_h = L.oh, out_w = L.ow;
    const int in_stride = L.in_stride, out_stride = L.out_stride;
    const int row_bytes = width * in_stride;
    const int rs = L.shift - 32;
    const int32_t rnd = 1 << (rs - 1);

    for (int oy = 0; oy < out_h; ++oy) {
        const int iy0 = oy * stride - pad;
        const int ky0 = iy0 < 0 ? -iy0 : 0;
        const int ky1 = iy0 + K > height ? height - iy0 : K;
        for (int ox = 0; ox < out_w; ++ox) {
            const int ix0 = ox * stride - pad;
            const int kx0 = ix0 < 0 ? -ix0 : 0;
            const int kx1 = ix0 + K > width ? width - ix0 : K;
            uint8_t* o = out + (oy * out_w + ox) * out_stride;
            const uint8_t* ip0 = in + ((iy0 + ky0) * width + ix0 + kx0) * in_stride;
            if (ky0 == 0 && ky1 == K && kx0 == 0 && kx1 == K) {
                dw_groups<IN_S, OUT_S, K>(ip0, row_bytes, in_stride, w, groups, bias, mult, o, 0, K, 0, K, rs, rnd);
            } else {
                dw_groups<IN_S, OUT_S, K>(ip0, row_bytes, in_stride, w, groups, bias, mult, o, ky0, ky1, kx0, kx1, rs, rnd);
            }
        }
    }
}

YOLO_HOT YOLO_NOINLINE void conv1_row(const Layer& L, const uint8_t* in, int cy, uint8_t* dst, const uint32_t* w,
                                      const int32_t* bias, const int32_t* mult) {
    const int height = L.h, width = L.w, conv_w = width / 2;
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = 2 * cy - 1 + ky;
        for (int cx = 0; cx < conv_w; ++cx) {
            uint8_t* d = g_im2col + cx * kConv1Taps + ky * 9;
            if (static_cast<unsigned>(iy) >= static_cast<unsigned>(height)) {
                std::memset(d, 0, 9);
            } else if (cx == 0) {
                std::memset(d, 0, 3);
                std::memcpy(d + 3, in + iy * width * 3, 6);
            } else {
                std::memcpy(d, in + (iy * width + 2 * cx - 1) * 3, 9);
            }
        }
    }
    for (int cx = 0; cx < conv_w; ++cx) {
        g_im2col[cx * kConv1Taps + 27] = 0;
    }
    pointwise<false, false>(conv_w, kConv1Taps, kConv1Channels, g_im2col, kConv1Taps, dst, kConv1Channels, w, bias, mult,
                            L.shift);
}

YOLO_HOT YOLO_NOINLINE void pool_rows(const uint8_t* const rows[3], int row_count, int conv_w, int pool_w,
                                      uint8_t* orow, int out_stride) {
    for (int px = 0; px < pool_w; ++px) {
        const int x0 = 2 * px - 1 < 0 ? 0 : 2 * px - 1;
        const int x1 = 2 * px + 1 > conv_w - 1 ? conv_w - 1 : 2 * px + 1;
        for (int c = 0; c < kConv1Channels; c += 4) {
            uint32_t m = 0;
            for (int r = 0; r < row_count; ++r) {
                for (int x = x0; x <= x1; ++x) {
                    const uint32_t v = ld32(rows[r] + x * kConv1Channels + c);
                    m = uqadd8(m, uqsub8(v, m));
                }
            }
            st32(orow + px * out_stride + c, m);
        }
    }
}

void conv1_pool(const Layer& L, const uint8_t* in, uint8_t* out, const uint32_t* w, const int32_t* bias,
                const int32_t* mult) {
    const int conv_h = L.h / 2, conv_w = L.w / 2;
    const int pool_h = conv_h / 2, pool_w = conv_w / 2;
    int computed = -1;
    for (int py = 0; py < pool_h; ++py) {
        const int y0 = 2 * py - 1 < 0 ? 0 : 2 * py - 1;
        const int y1 = 2 * py + 1 > conv_h - 1 ? conv_h - 1 : 2 * py + 1;
        while (computed < y1) {
            ++computed;
            conv1_row(L, in, computed, g_ring[computed % 3], w, bias, mult);
        }
        const uint8_t* rows[3];
        int count = 0;
        for (int y = y0; y <= y1; ++y) rows[count++] = g_ring[y % 3];
        pool_rows(rows, count, conv_w, pool_w, out + py * pool_w * L.out_stride, L.out_stride);
    }
}

void split_copy(const Layer& L, const uint8_t* in, uint8_t* even_out, uint8_t* odd_out) {
    const int npix = L.h * L.w;
    const int half = L.cin / 2;
    const int32_t lo = L.out_signed ? -128 : 0, hi = L.out_signed ? 127 : 255;
    for (int p = 0; p < npix; ++p) {
        const uint8_t* s = in + p * L.in_stride;
        uint8_t* e = even_out + p * L.out_stride;
        uint8_t* o = odd_out + p * L.out2_stride;
        for (int c = 0; c < half; ++c) {
            o[c] = s[2 * c + 1];
            if (L.identity) {
                e[c] = s[2 * c];
            } else {
                const int32_t v = L.in_signed ? static_cast<int8_t>(s[2 * c]) : s[2 * c];
                e[c] = static_cast<uint8_t>(requant_generic(v, L.mult, L.shift, lo, hi));
            }
        }
    }
}

void upsample_copy(const Layer& L, const uint8_t* in, uint8_t* out) {
    const int32_t lo = L.out_signed ? -128 : 0, hi = L.out_signed ? 127 : 255;
    for (int y = 0; y < L.oh; ++y) {
        for (int x = 0; x < L.ow; ++x) {
            const uint8_t* s = in + ((y >> 1) * L.w + (x >> 1)) * L.in_stride;
            uint8_t* d = out + (y * L.ow + x) * L.out_stride;
            for (int c = 0; c < L.cin; ++c) {
                if (L.identity) {
                    d[c] = s[c];
                } else {
                    const int32_t v = L.in_signed ? static_cast<int8_t>(s[c]) : s[c];
                    d[c] = static_cast<uint8_t>(requant_generic(v, L.mult, L.shift, lo, hi));
                }
            }
        }
    }
}

uint32_t view_checksum(const uint8_t* base, int pixels, int channels, int stride) {
    uint32_t s1 = 0, s2 = 0;
    for (int p = 0; p < pixels; ++p) {
        const uint8_t* q = base + p * stride;
        for (int c = 0; c < channels; ++c) {
            s1 += q[c];
            s2 += s1;
        }
    }
    return s2 + 0x9E3779B1u * s1;
}

uint32_t layer_checksum(const Layer& L, const uint8_t* arena) {
    const int pixels = L.oh * L.ow;
    uint32_t value = view_checksum(arena + L.out_off, pixels, L.cout, L.out_stride);
    if (L.kind == kSplit) {
        value = value * kHashPrime + view_checksum(arena + L.out2_off, pixels, L.cout, L.out2_stride);
    }
    return value;
}

template <bool IN_S, bool OUT_S>
void run_depthwise(const Layer& L, const uint8_t* in, uint8_t* out, const int32_t* bias, const int32_t* mult) {
    if (L.k == 3) {
        depthwise<IN_S, OUT_S, 3>(L, in, out, g_weights, bias, mult);
    } else {
        depthwise<IN_S, OUT_S, 5>(L, in, out, g_weights, bias, mult);
    }
}

template <bool IN_S, bool OUT_S>
void run_pointwise(const Layer& L, const uint8_t* in, uint8_t* out, const int32_t* bias, const int32_t* mult) {
    pointwise<IN_S, OUT_S>(L.h * L.w, L.cin, L.cout, in, L.in_stride, out, L.out_stride, g_weights, bias, mult, L.shift);
}

}  // namespace

void run_network(uint8_t* arena, uint32_t* layer_us, uint32_t* layer_sum) {
    for (int i = 0; i < kLayerCount; ++i) {
        const Layer& L = kLayers[i];
        const uint32_t started = yolo_now_us();
        const uint8_t* in = arena + L.in_off;
        uint8_t* out = arena + L.out_off;
        const int32_t* bias = kBias + L.b_off;
        const int32_t* mult = kMult + L.b_off;
        if (L.w_len) {
            std::memcpy(g_weights, kWeights + L.w_off, L.w_len);
        }
        switch (L.kind) {
            case kConv1Pool:
                conv1_pool(L, in, out, g_weights, bias, mult);
                break;
            case kPointwise:
                if (L.in_signed) {
                    if (L.out_signed) run_pointwise<true, true>(L, in, out, bias, mult);
                    else run_pointwise<true, false>(L, in, out, bias, mult);
                } else {
                    if (L.out_signed) run_pointwise<false, true>(L, in, out, bias, mult);
                    else run_pointwise<false, false>(L, in, out, bias, mult);
                }
                break;
            case kDepthwise:
                if (L.in_signed) {
                    if (L.out_signed) run_depthwise<true, true>(L, in, out, bias, mult);
                    else run_depthwise<true, false>(L, in, out, bias, mult);
                } else {
                    if (L.out_signed) run_depthwise<false, true>(L, in, out, bias, mult);
                    else run_depthwise<false, false>(L, in, out, bias, mult);
                }
                break;
            case kSplit:
                split_copy(L, in, out, arena + L.out2_off);
                break;
            case kUpCopy:
                upsample_copy(L, in, out);
                break;
        }
        if (layer_us) {
            layer_us[i] = yolo_now_us() - started;
        }
        if (layer_sum) {
            layer_sum[i] = layer_checksum(L, arena);
        }
    }
}

uint32_t output_checksum(const uint8_t* arena) {
    uint32_t value = 0;
    for (const Head& head : kHeads) {
        const int pixels = head.h * head.w;
        value = value * kHashPrime + view_checksum(arena + head.reg_off, pixels, 4 * kNumAnchors, 4 * kNumAnchors);
        value = value * kHashPrime + view_checksum(arena + head.obj_off, pixels, kNumAnchors, kNumAnchors);
        value = value * kHashPrime + view_checksum(arena + head.cls_off, pixels, kNumClasses, kNumClasses);
    }
    return value;
}

}  // namespace yolo
