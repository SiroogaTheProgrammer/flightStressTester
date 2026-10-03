#include <cstdio>
#include <cstdlib>
#include <vector>

#include "yolo_decode.h"
#include "yolo_engine.h"
#include "yolo_model_data.h"

// usage: yolo_host_test <raw_rgb_file> [conf_permil] [iou_permil]
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    std::vector<uint8_t> arena(yolo::kArenaSize + 64, 0);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(arena.data() + yolo::kInputOffset, 1, yolo::kInputBytes, f) != static_cast<size_t>(yolo::kInputBytes)) {
        std::fprintf(stderr, "cannot read %d bytes from %s\n", yolo::kInputBytes, argv[1]);
        return 3;
    }
    std::fclose(f);
    const float conf = (argc > 2 ? std::atoi(argv[2]) : 300) / 1000.0f;
    const float iou = (argc > 3 ? std::atoi(argv[3]) : 400) / 1000.0f;

    uint32_t sums[yolo::kLayerCount];
    uint32_t us[yolo::kLayerCount];
    yolo::run_network(arena.data(), us, sums);
    std::printf("SUMS");
    for (int i = 0; i < yolo::kLayerCount; ++i) std::printf(",%u", sums[i]);
    std::printf("\nOUT,%u\n", yolo::output_checksum(arena.data()));
    yolo::Detection dets[32];
    const int n = yolo::decode_detections(arena.data(), conf, iou, dets, 32);
    for (int i = 0; i < n; ++i) {
        std::printf("DET,%d,%u,%d,%d,%d,%d\n", dets[i].cls, dets[i].score_permil, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    }
    return 0;
}
