#pragma once

#include <cstdint>

namespace yolo {

enum LayerKind : uint8_t { kConv1Pool = 0, kPointwise = 1, kDepthwise = 2, kSplit = 3, kUpCopy = 4 };

struct Layer {
    uint8_t kind, k, stride, in_signed, out_signed, identity, shift, pad0;
    uint16_t h, w, oh, ow, cin, cout, in_stride, out_stride, out2_stride, pad1;
    uint32_t in_off, out_off, out2_off, w_off, w_len, b_off;
    int32_t mult;
};

struct Head {
    uint32_t reg_off, obj_off, cls_off;
    float reg_scale, obj_scale, cls_scale;
    uint16_t h, w;
    float anchors[6];
};

struct Detection {
    int16_t x1, y1, x2, y2;
    uint16_t score_permil;
    uint8_t cls;
    uint8_t pad;
};

}  // namespace yolo
