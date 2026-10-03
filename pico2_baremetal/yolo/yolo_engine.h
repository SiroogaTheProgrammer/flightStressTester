#pragma once

#include <cstdint>

namespace yolo {

void run_network(uint8_t* arena, uint32_t* layer_us, uint32_t* layer_sum);
uint32_t output_checksum(const uint8_t* arena);

}  // namespace yolo
