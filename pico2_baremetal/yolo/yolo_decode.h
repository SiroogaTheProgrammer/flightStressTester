#pragma once

#include <cstdint>

#include "yolo_layer.h"

namespace yolo {

int decode_detections(const uint8_t* arena, float conf, float iou, Detection* out, int max_out);

}  // namespace yolo
