add_executable(lab3
    ${CMAKE_SOURCE_DIR}/common/source/main.cpp
    source/application.cpp
    source/camera.hpp
    source/lights.hpp
    source/geometry.hpp
    source/geometry.cpp
    source/texture.hpp
    source/texture.cpp
)

target_include_directories(lab3 PRIVATE source)
target_link_libraries(lab3 PRIVATE mai_framework)

if(WIN32)
    if(MSVC)
        target_link_options(lab3 PRIVATE /subsystem:windows /entry:mainCRTStartup)
    else()
        target_link_options(lab3 PRIVATE -mwindows)
    endif()
endif()

# Recompile shaders when modified (if glslc is installed)
add_lab_shaders(lab3 ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    shader.vert
    shader.frag
    unlit.vert
    unlit.frag
)

# Copy shaders directory next to executable so it works regardless of working directory
add_custom_command(TARGET lab3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:lab3>/shaders
)

# Copy textures directory next to executable so images load from anywhere
add_custom_command(TARGET lab3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/textures
        $<TARGET_FILE_DIR:lab3>/textures
)
