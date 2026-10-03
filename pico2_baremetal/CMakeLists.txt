# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(pico2_baremetal C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

add_executable(pico2_baremetal pico2_baremetal.cpp )

pico_set_program_name(pico2_baremetal "pico2_baremetal")
pico_set_program_version(pico2_baremetal "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(pico2_baremetal 0)
pico_enable_stdio_usb(pico2_baremetal 1)

# Add the standard library to the build
target_link_libraries(pico2_baremetal
    pico_stdlib
    pico_multicore
    pico_util)

# Add the standard include files to the build
target_include_directories(pico2_baremetal PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

# Add any user requested libraries
target_link_libraries(pico2_baremetal 
        
        )

pico_add_extra_outputs(pico2_baremetal)

# Yolo-FastestV2 int8 detector, separate firmware: core 1 runs inference, core 0 handles USB
set(YOLO_OVERCLOCK_MHZ 0 CACHE STRING "Optional system clock in MHz for pico2_yolo (0 keeps the default)")
option(YOLO_KERNELS_IN_RAM "Run hot YOLO kernels from SRAM" ON)

add_executable(pico2_yolo
        yolo/yolo_main.cpp
        yolo/yolo_engine.cpp
        yolo/yolo_decode.cpp
        yolo/yolo_model_data.cpp
)

pico_set_program_name(pico2_yolo "pico2_yolo")
pico_set_program_version(pico2_yolo "0.1")
pico_enable_stdio_uart(pico2_yolo 0)
pico_enable_stdio_usb(pico2_yolo 1)

target_include_directories(pico2_yolo PRIVATE ${CMAKE_CURRENT_LIST_DIR}/yolo)
if(YOLO_KERNELS_IN_RAM)
    target_compile_definitions(pico2_yolo PRIVATE YOLO_KERNELS_IN_RAM=1)
endif()
target_link_libraries(pico2_yolo
        pico_stdlib
        pico_multicore
        pico_util
        hardware_vreg
)
if(YOLO_OVERCLOCK_MHZ GREATER 0)
    target_compile_definitions(pico2_yolo PRIVATE YOLO_OVERCLOCK_MHZ=${YOLO_OVERCLOCK_MHZ})
endif()

pico_add_extra_outputs(pico2_yolo)

