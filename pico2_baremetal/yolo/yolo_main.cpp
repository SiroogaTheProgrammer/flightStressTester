#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hardware/clocks.h"
#include "pico/multicore.h"
#include "pico/stdio/driver.h"
#include "pico/stdio_usb.h"
#include "pico/stdlib.h"
#include "pico/util/queue.h"

#ifdef YOLO_OVERCLOCK_MHZ
#include "hardware/vreg.h"
#endif

#include "yolo_decode.h"
#include "yolo_engine.h"
#include "yolo_model_data.h"

namespace {

using namespace yolo;

constexpr int kMaxDetections = 32;
constexpr uint32_t kSlotCount = 1;
constexpr uint32_t kRxTimeoutUs = 3000000;

struct Slot {
    uint32_t id;
    uint32_t rx_us;
    alignas(4) uint8_t pixels[kInputBytes];
};

struct Result {
    uint32_t id, rx_us, infer_us, decode_us, avg_us, min_us, max_us, completed, checksum, count;
    Detection dets[kMaxDetections];
};

alignas(16) uint8_t g_arena[kArenaSize];
Slot g_slots[kSlotCount];
alignas(8) uint32_t g_core1_stack[2048];
queue_t g_free_q, g_full_q, g_result_q;
uint32_t g_profile_us[kLayerCount];
uint32_t g_profile_sum[kLayerCount];
Result g_core1_result;
Result g_core0_result;
volatile uint32_t g_conf_permil = 300;
volatile uint32_t g_iou_permil = 400;
volatile bool g_debug_sums = false;
volatile bool g_reset_stats = false;
volatile uint32_t g_core1_phase = 0;

void core1_main() {
    uint64_t total_us = 0;
    uint32_t completed = 0, min_us = 0xFFFFFFFFu, max_us = 0;
    while (true) {
        uint32_t index;
        g_core1_phase = 1;
        queue_remove_blocking(&g_full_q, &index);
        g_core1_phase = 2;
        const Slot& slot = g_slots[index];
        std::memcpy(g_arena + kInputOffset, slot.pixels, kInputBytes);
        g_core1_result.id = slot.id;
        g_core1_result.rx_us = slot.rx_us;
        queue_add_blocking(&g_free_q, &index);

        if (g_reset_stats) {
            total_us = 0;
            completed = 0;
            min_us = 0xFFFFFFFFu;
            max_us = 0;
            g_reset_stats = false;
        }

        const uint32_t t0 = time_us_32();
        g_core1_phase = 3;
        run_network(g_arena, g_profile_us, g_debug_sums ? g_profile_sum : nullptr);
        const uint32_t t1 = time_us_32();
        g_core1_phase = 4;
        g_core1_result.checksum = output_checksum(g_arena);
        g_core1_phase = 5;
        g_core1_result.count = static_cast<uint32_t>(decode_detections(
            g_arena, g_conf_permil / 1000.0f, g_iou_permil / 1000.0f, g_core1_result.dets, kMaxDetections));
        const uint32_t t2 = time_us_32();

        const uint32_t infer_us = t1 - t0;
        ++completed;
        total_us += infer_us;
        min_us = infer_us < min_us ? infer_us : min_us;
        max_us = infer_us > max_us ? infer_us : max_us;
        g_core1_result.infer_us = infer_us;
        g_core1_result.decode_us = t2 - t1;
        g_core1_result.avg_us = static_cast<uint32_t>(total_us / completed);
        g_core1_result.min_us = min_us;
        g_core1_result.max_us = max_us;
        g_core1_result.completed = completed;
        g_core1_phase = 6;
        queue_add_blocking(&g_result_q, &g_core1_result);
    }
}

unsigned long ul(uint32_t v) { return static_cast<unsigned long>(v); }

void print_ready() {
    printf("READY,YOLO,%d,%lu,%d,%lu,%lu,%lu\n", kInputSize, ul(kArenaSize), kLayerCount,
           ul(clock_get_hz(clk_sys) / 1000000), ul(kSlotCount), ul(kMaxDetections));
}

void print_result(const Result& r) {
    for (uint32_t i = 0; i < r.count; ++i) {
        const Detection& d = r.dets[i];
        printf("DET,%lu,%u,%u,%d,%d,%d,%d\n", ul(r.id), d.cls, d.score_permil, d.x1, d.y1, d.x2, d.y2);
    }
    printf("RESULT,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%08lx\n", ul(r.id), ul(r.count), ul(r.infer_us),
           ul(r.decode_us), ul(r.rx_us), ul(r.avg_us), ul(r.min_us), ul(r.max_us), ul(r.completed), ul(r.checksum));
}

void print_array(const char* tag, const uint32_t* values, bool hex) {
    printf("%s,%d", tag, kLayerCount);
    for (int i = 0; i < kLayerCount; ++i) printf(hex ? ",%08lx" : ",%lu", ul(values[i]));
    printf("\n");
}

enum class State { Line, WaitSlot, Payload };

State g_state = State::Line;
char g_line[48];
int g_line_len = 0;
uint32_t g_pending_id = 0, g_slot_index = 0, g_received = 0, g_rx_start = 0, g_last_rx = 0;

bool starts_with(const char* s, const char* prefix) { return std::strncmp(s, prefix, std::strlen(prefix)) == 0; }

void handle_line(const char* line) {
    if (std::strcmp(line, "HELLO") == 0) {
        print_ready();
    } else if (std::strcmp(line, "STATUS") == 0) {
        printf("STATUS,%lu,%lu,%lu,%lu\n", ul(g_core1_phase),
               ul(queue_get_level(&g_full_q)), ul(queue_get_level(&g_free_q)),
               ul(queue_get_level(&g_result_q)));
    } else if (starts_with(line, "IMG,")) {
        g_pending_id = static_cast<uint32_t>(std::strtoul(line + 4, nullptr, 10));
        g_state = State::WaitSlot;
    } else if (starts_with(line, "CFG,")) {
        char* end = nullptr;
        const unsigned long conf = std::strtoul(line + 4, &end, 10);
        const unsigned long iou = *end == ',' ? std::strtoul(end + 1, nullptr, 10) : g_iou_permil;
        g_conf_permil = conf > 1000 ? 1000 : conf;
        g_iou_permil = iou > 1000 ? 1000 : iou;
        printf("OK,CFG,%lu,%lu\n", ul(g_conf_permil), ul(g_iou_permil));
    } else if (std::strcmp(line, "PROFILE") == 0) {
        print_array("PROFILE", g_profile_us, false);
    } else if (std::strcmp(line, "SUMS") == 0) {
        print_array("SUMS", g_profile_sum, true);
    } else if (starts_with(line, "DEBUG,")) {
        g_debug_sums = line[6] == '1';
        printf("OK,DEBUG,%d\n", g_debug_sums ? 1 : 0);
    } else if (std::strcmp(line, "STATS,RESET") == 0) {
        g_reset_stats = true;
        printf("OK,STATS,RESET\n");
    } else {
        printf("ERR,unknown_command\n");
    }
}

void service_input() {
    switch (g_state) {
        case State::Line: {
            char c;
            if (stdio_usb.in_chars(&c, 1) != 1) {
                tight_loop_contents();
                break;
            }
            if (c == '\n' || c == '\r') {
                if (g_line_len > 0) {
                    g_line[g_line_len] = '\0';
                    g_line_len = 0;
                    handle_line(g_line);
                }
            } else if (g_line_len < static_cast<int>(sizeof(g_line)) - 1) {
                g_line[g_line_len++] = c;
            }
            break;
        }
        case State::WaitSlot:
            if (queue_try_remove(&g_free_q, &g_slot_index)) {
                g_received = 0;
                g_rx_start = g_last_rx = time_us_32();
                g_state = State::Payload;
            }
            break;
        case State::Payload: {
            uint8_t* dst = g_slots[g_slot_index].pixels;
            const int n = stdio_usb.in_chars(reinterpret_cast<char*>(dst + g_received),
                                             static_cast<int>(kInputBytes - g_received));
            const uint32_t now = time_us_32();
            if (n > 0) {
                g_received += static_cast<uint32_t>(n);
                g_last_rx = now;
            } else if (now - g_last_rx > kRxTimeoutUs) {
                printf("ERR,%lu,rx_timeout,%lu\n", ul(g_pending_id), ul(g_received));
                queue_try_add(&g_free_q, &g_slot_index);
                g_state = State::Line;
                break;
            } else {
                tight_loop_contents();
            }
            if (g_received == static_cast<uint32_t>(kInputBytes)) {
                g_slots[g_slot_index].id = g_pending_id;
                g_slots[g_slot_index].rx_us = now - g_rx_start;
                queue_try_add(&g_full_q, &g_slot_index);
                printf("RECEIVED,%lu,%lu\n", ul(g_pending_id), ul(g_received));
                g_state = State::Line;
            }
            break;
        }
    }
}

}  // namespace

int main() {
#ifdef YOLO_OVERCLOCK_MHZ
    vreg_set_voltage(VREG_VOLTAGE_1_30);
    sleep_ms(2);
    set_sys_clock_khz(YOLO_OVERCLOCK_MHZ * 1000, true);
#endif
    stdio_init_all();
    queue_init(&g_free_q, sizeof(uint32_t), kSlotCount);
    queue_init(&g_full_q, sizeof(uint32_t), kSlotCount);
    queue_init(&g_result_q, sizeof(Result), 3);
    for (uint32_t i = 0; i < kSlotCount; ++i) queue_try_add(&g_free_q, &i);
    multicore_launch_core1_with_stack(core1_main, g_core1_stack, sizeof(g_core1_stack));

    while (true) {
        while (queue_try_remove(&g_result_q, &g_core0_result)) {
            print_result(g_core0_result);
        }
        service_input();
    }
}
