add_executable(lab2
    ${CMAKE_SOURCE_DIR}/common/source/main.cpp
    source/application.cpp
    source/camera.hpp
    source/lights.hpp
    source/geometry.hpp
    source/geometry.cpp
)

target_include_directories(lab2 PRIVATE source)
target_link_libraries(lab2 PRIVATE mai_framework)

if(WIN32)
    if(MSVC)
        target_link_options(lab2 PRIVATE /subsystem:windows /entry:mainCRTStartup)
    else()
        target_link_options(lab2 PRIVATE -mwindows)
    endif()
endif()

# Recompile shaders when modified (if glslc is installed)
add_lab_shaders(lab2 ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    shader.vert
    shader.frag
    unlit.vert
    unlit.frag
)

# Copy shaders directory next to executable so it works regardless of working directory
add_custom_command(TARGET lab2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:lab2>/shaders
)
