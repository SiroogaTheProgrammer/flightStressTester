#include <cmath>
#include <cstdint>
#include <cstdio>

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"

#include "digit_model.h"

namespace {

constexpr uint32_t kImageSize = 28 * 28;
constexpr uint32_t kHiddenSize = 32;
constexpr uint32_t kClassCount = 10;
constexpr uint32_t kQueueCapacity = 8;

struct ImageFrame {
    uint32_t sequence;
    uint8_t pixels[kImageSize];
};

struct InferenceResult {
    uint32_t sequence;
    uint32_t latency_us;
    uint32_t average_us;
    uint32_t maximum_us;
    uint32_t completed;
    uint16_t confidence_permil;
    uint8_t digit;
};

queue_t image_queue;
queue_t result_queue;

InferenceResult infer(const ImageFrame& frame) {
    const uint64_t start_us = time_us_64();
    float hidden[kHiddenSize];

    for (uint32_t neuron = 0; neuron < kHiddenSize; ++neuron) {
        float sum = kHiddenBiases[neuron];
        const float* weights = &kLayer1Weights[neuron * kImageSize];
        for (uint32_t pixel = 0; pixel < kImageSize; ++pixel) {
            sum += weights[pixel] * (static_cast<float>(frame.pixels[pixel]) / 255.0f);
        }
        hidden[neuron] = sum > 0.0f ? sum : 0.0f;
    }

    float logits[kClassCount];
    float maximum_logit = -INFINITY;
    for (uint32_t digit = 0; digit < kClassCount; ++digit) {
        float sum = kOutputBiases[digit];
        const float* weights = &kLayer2Weights[digit * kHiddenSize];
        for (uint32_t neuron = 0; neuron < kHiddenSize; ++neuron) {
            sum += weights[neuron] * hidden[neuron];
        }
        logits[digit] = sum;
        if (sum > maximum_logit) {
            maximum_logit = sum;
        }
    }

    float exponent_sum = 0.0f;
    uint32_t prediction = 0;
    float best_probability = -1.0f;
    for (uint32_t digit = 0; digit < kClassCount; ++digit) {
        const float probability = expf(logits[digit] - maximum_logit);
        logits[digit] = probability;
        exponent_sum += probability;
    }
    for (uint32_t digit = 0; digit < kClassCount; ++digit) {
        const float probability = logits[digit] / exponent_sum;
        if (probability > best_probability) {
            best_probability = probability;
            prediction = digit;
        }
    }

    const uint32_t latency_us = static_cast<uint32_t>(time_us_64() - start_us);
    static uint64_t total_latency_us = 0;
    static uint32_t completed = 0;
    static uint32_t maximum_latency_us = 0;
    ++completed;
    total_latency_us += latency_us;
    if (latency_us > maximum_latency_us) {
        maximum_latency_us = latency_us;
    }

    return {
        frame.sequence,
        latency_us,
        static_cast<uint32_t>(total_latency_us / completed),
        maximum_latency_us,
        completed,
        static_cast<uint16_t>(best_probability * 1000.0f + 0.5f),
        static_cast<uint8_t>(prediction),
    };
}

void inference_core() {
    while (true) {
        ImageFrame frame;
        if (!queue_try_remove(&image_queue, &frame)) {
            __wfe();
            continue;
        }

        const InferenceResult result = infer(frame);
        queue_try_add(&result_queue, &result);
    }
}

bool read_image(ImageFrame& frame) {
    for (uint32_t pixel = 0; pixel < kImageSize; ++pixel) {
        int value = getchar_timeout_us(1000000);
        if (value < 0) {
            return false;
        }
        frame.pixels[pixel] = static_cast<uint8_t>(value);
    }
    return true;
}

void print_result(const InferenceResult& result) {
    printf("RESULT,%lu,%u,%u.%u,%lu,%lu,%lu,%lu\n",
           static_cast<unsigned long>(result.sequence),
           result.digit,
           result.confidence_permil / 10,
           result.confidence_permil % 10,
           static_cast<unsigned long>(result.latency_us),
           static_cast<unsigned long>(result.average_us),
           static_cast<unsigned long>(result.maximum_us),
           static_cast<unsigned long>(result.completed));
}

}  // namespace

int main() {
    stdio_init_all();
    queue_init(&image_queue, sizeof(ImageFrame), kQueueCapacity);
    queue_init(&result_queue, sizeof(InferenceResult), kQueueCapacity);
    multicore_launch_core1(inference_core);

    printf("READY,28,28,784\n");
    uint32_t next_sequence = 1;
    char command[6] = {};
    uint32_t command_length = 0;

    while (true) {
        InferenceResult result;
        while (queue_try_remove(&result_queue, &result)) {
            print_result(result);
        }

        const int value = getchar_timeout_us(0);
        if (value < 0) {
            tight_loop_contents();
            continue;
        }

        if (value == '\n' || value == '\r') {
            if (command_length == 5 && command[0] == 'H' && command[1] == 'E' && command[2] == 'L' &&
                command[3] == 'L' && command[4] == 'O') {
                printf("READY,28,28,784\n");
            } else if (command_length == 3 && command[0] == 'I' && command[1] == 'M' && command[2] == 'G') {
                ImageFrame frame{};
                frame.sequence = next_sequence++;
                if (!read_image(frame)) {
                    printf("ERROR,%lu,incomplete_image\n", static_cast<unsigned long>(frame.sequence));
                } else if (!queue_try_add(&image_queue, &frame)) {
                    printf("DROP,%lu,input_queue_full\n", static_cast<unsigned long>(frame.sequence));
                } else {
                    __sev();
                }
            }
            command_length = 0;
        } else if (command_length < sizeof(command)) {
            command[command_length++] = static_cast<char>(value);
        } else {
            command_length = 0;
        }
    }
}