#include "yolo_decode.h"

#include <cmath>

#include "yolo_model_data.h"

namespace yolo {
namespace {

constexpr int kMaxCandidates = 192;

struct Candidate {
    float x1, y1, x2, y2, score;
    int cls;
};

Candidate g_candidates[kMaxCandidates];

inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

int add_candidate(int count, const Candidate& c) {
    if (count < kMaxCandidates) {
        g_candidates[count] = c;
        return count + 1;
    }
    int lowest = 0;
    for (int i = 1; i < count; ++i) {
        if (g_candidates[i].score < g_candidates[lowest].score) lowest = i;
    }
    if (c.score > g_candidates[lowest].score) g_candidates[lowest] = c;
    return count;
}

bool overlaps(const Candidate& a, const Candidate& b, float iou) {
    const float iw = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1);
    const float ih = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1);
    if (iw <= 0.0f || ih <= 0.0f) return false;
    const float inter = iw * ih;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / uni > iou;
}

int16_t to_i16(float v) {
    v = v < -32000.0f ? -32000.0f : (v > 32000.0f ? 32000.0f : v);
    return static_cast<int16_t>(lroundf(v));
}

}  // namespace

int decode_detections(const uint8_t* arena, float conf, float iou, Detection* out, int max_out) {
    int count = 0;
    for (const Head& head : kHeads) {
        const float stride = static_cast<float>(kInputSize) / head.h;
        const int8_t* reg = reinterpret_cast<const int8_t*>(arena + head.reg_off);
        const int8_t* obj = reinterpret_cast<const int8_t*>(arena + head.obj_off);
        const int8_t* cls = reinterpret_cast<const int8_t*>(arena + head.cls_off);
        for (int y = 0; y < head.h; ++y) {
            for (int x = 0; x < head.w; ++x) {
                const int cell = y * head.w + x;
                float objectness[kNumAnchors];
                bool any = false;
                for (int a = 0; a < kNumAnchors; ++a) {
                    objectness[a] = sigmoid(obj[cell * kNumAnchors + a] * head.obj_scale);
                    any = any || objectness[a] > conf;
                }
                if (!any) continue;

                const int8_t* logits = cls + cell * kNumClasses;
                int best = 0;
                int8_t peak = logits[0];
                for (int i = 1; i < kNumClasses; ++i) {
                    if (logits[i] > peak) {
                        peak = logits[i];
                        best = i;
                    }
                }
                float denom = 0.0f;
                for (int i = 0; i < kNumClasses; ++i) {
                    denom += expf((logits[i] - peak) * head.cls_scale);
                }
                const float class_prob = 1.0f / denom;

                for (int a = 0; a < kNumAnchors; ++a) {
                    const float score = objectness[a] * class_prob;
                    if (score <= conf) continue;
                    const int8_t* r = reg + cell * 4 * kNumAnchors + a * 4;
                    const float cx = (sigmoid(r[0] * head.reg_scale) * 2.0f - 0.5f + x) * stride;
                    const float cy = (sigmoid(r[1] * head.reg_scale) * 2.0f - 0.5f + y) * stride;
                    const float sw = sigmoid(r[2] * head.reg_scale) * 2.0f;
                    const float sh = sigmoid(r[3] * head.reg_scale) * 2.0f;
                    const float bw = sw * sw * head.anchors[a * 2];
                    const float bh = sh * sh * head.anchors[a * 2 + 1];
                    count = add_candidate(count, {cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f, score, best});
                }
            }
        }
    }

    for (int i = 1; i < count; ++i) {
        const Candidate key = g_candidates[i];
        int j = i - 1;
        while (j >= 0 && g_candidates[j].score < key.score) {
            g_candidates[j + 1] = g_candidates[j];
            --j;
        }
        g_candidates[j + 1] = key;
    }

    int kept = 0;
    for (int i = 0; i < count && kept < max_out; ++i) {
        bool suppressed = false;
        for (int j = 0; j < i && !suppressed; ++j) {
            if (g_candidates[j].score < 0.0f) continue;
            suppressed = g_candidates[j].cls == g_candidates[i].cls && overlaps(g_candidates[i], g_candidates[j], iou);
        }
        if (suppressed) {
            g_candidates[i].score = -1.0f;
            continue;
        }
        const Candidate& c = g_candidates[i];
        out[kept++] = {to_i16(c.x1), to_i16(c.y1), to_i16(c.x2), to_i16(c.y2),
                       static_cast<uint16_t>(c.score * 1000.0f + 0.5f), static_cast<uint8_t>(c.cls), 0};
    }
    return kept;
}

}  // namespace yolo
