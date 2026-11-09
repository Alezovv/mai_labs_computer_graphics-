add_executable(lab1
    ${CMAKE_SOURCE_DIR}/common/source/main.cpp
    source/application.cpp
    source/torus.cpp
)

target_include_directories(lab1 PRIVATE source)
target_link_libraries(lab1 PRIVATE mai_framework)

if(WIN32)
    if(MSVC)
        target_link_options(lab1 PRIVATE /subsystem:windows /entry:mainCRTStartup)
    else()
        target_link_options(lab1 PRIVATE -mwindows)
    endif()
endif()

# Recompile shaders when modified (if glslc is installed)
add_lab_shaders(lab1 ${CMAKE_CURRENT_SOURCE_DIR}/shaders shader.vert shader.frag)

# Copy shaders directory next to executable so it works regardless of working directory
add_custom_command(TARGET lab1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:lab1>/shaders
)

# Compatibility alias for starter app target name
add_custom_target(vulkan-starter-app DEPENDS lab1)
