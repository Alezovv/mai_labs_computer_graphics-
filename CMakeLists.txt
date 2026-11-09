cmake_minimum_required(VERSION 3.20)
include(FetchContent)

project(mai-computer-graphics LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# -----------------------------------------------------------------------------
# Third-party dependencies
# -----------------------------------------------------------------------------
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.5.1
)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG v1.4.307
)

FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)

find_package(Vulkan REQUIRED)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)

FetchContent_MakeAvailable(
    glfw
    vk-bootstrap
    VulkanMemoryAllocator
    imgui
    glm
)

# -----------------------------------------------------------------------------
# Common framework library
# -----------------------------------------------------------------------------
add_library(mai_framework STATIC
    common/source/graphics_internal.cpp
)

target_include_directories(mai_framework PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(mai_framework PUBLIC
    glfw
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
)

target_sources(mai_framework PRIVATE
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
)

target_compile_definitions(mai_framework PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR LINUX)
    target_compile_options(mai_framework PUBLIC -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
elseif(APPLE)
    target_compile_options(mai_framework PUBLIC -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
elseif(WIN32)
    if(MSVC)
        target_compile_options(mai_framework PUBLIC /W4 /wd4201 /wd4100)
    else()
        target_compile_options(mai_framework PUBLIC -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter)
    endif()
    target_compile_definitions(mai_framework PUBLIC _USE_MATH_DEFINES)
endif()

# -----------------------------------------------------------------------------
# Shader compilation helper
# -----------------------------------------------------------------------------
find_program(GLSLC_FOUND glslc)

function(add_lab_shaders TARGET_NAME SHADER_DIR)
    if(GLSLC_FOUND)
        set(SHADER_OUTPUTS)
        foreach(SHADER_FILE ${ARGN})
            set(SHADER_SRC "${SHADER_DIR}/${SHADER_FILE}")
            set(SHADER_SPV "${SHADER_DIR}/${SHADER_FILE}.spv")
            add_custom_command(
                OUTPUT "${SHADER_SPV}"
                COMMAND glslc "${SHADER_SRC}" -o "${SHADER_SPV}"
                DEPENDS "${SHADER_SRC}"
                COMMENT "Compiling ${SHADER_FILE} to ${SHADER_FILE}.spv"
            )
            list(APPEND SHADER_OUTPUTS "${SHADER_SPV}")
        endforeach()
        add_custom_target(${TARGET_NAME}_shaders DEPENDS ${SHADER_OUTPUTS})
        add_dependencies(${TARGET_NAME} ${TARGET_NAME}_shaders)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Labs
# -----------------------------------------------------------------------------
add_subdirectory(lab1)
add_subdirectory(lab2)
add_subdirectory(lab3)
