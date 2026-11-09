#pragma once

#include "graphics_internal.hpp"

namespace application {

const char* getTitle();

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application
