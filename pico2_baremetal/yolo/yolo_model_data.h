#pragma once

#include "yolo_layer.h"

namespace yolo {

constexpr int kInputSize = 224;
constexpr int kInputBytes = 150528;
constexpr uint32_t kInputOffset = 0;
constexpr uint32_t kArenaSize = 225792;
constexpr int kLayerCount = 93;
constexpr uint32_t kMaxWeightBytes = 41472;
constexpr int kNumClasses = 80;
constexpr int kNumAnchors = 3;

extern const Layer kLayers[kLayerCount];
extern const Head kHeads[2];
extern const int8_t kWeights[];
extern const int32_t kBias[];
extern const int32_t kMult[];

}  // namespace yolo
