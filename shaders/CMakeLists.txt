cmake_minimum_required(VERSION 3.5.1)

set(DEPS)
set(GLSLANG_VALIDATOR "$ENV{VULKAN_SDK}/bin/glslangValidator")

macro(build_shader src dst)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/shaders/${dst}
        COMMAND ${GLSLANG_VALIDATOR} -s -V -o ${CMAKE_BINARY_DIR}/shaders/${dst} ${PROJECT_SOURCE_DIR}/shaders/${src}
        DEPENDS ${PROJECT_SOURCE_DIR}/shaders/${src}
        )
    set(DEPS ${DEPS} ${CMAKE_BINARY_DIR}/shaders/${dst})
endmacro(build_shader)

build_shader("shader.comp" "comp.spv")

add_custom_target(generate-shaders DEPENDS ${DEPS})
