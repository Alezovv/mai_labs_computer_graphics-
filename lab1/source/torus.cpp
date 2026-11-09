#include "torus.hpp"

#include <cmath>
#include <algorithm>

namespace graphics {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
} // namespace

TorusMesh TorusMesh::generate(float majorRadius, float minorRadius,
                              uint32_t radialSegments, uint32_t tubularSegments,
                              int colorMode) {
    TorusMesh mesh;

    radialSegments = std::max(radialSegments, 3u);
    tubularSegments = std::max(tubularSegments, 3u);

    const float totalRadius = majorRadius + minorRadius;

    mesh.vertices.reserve((radialSegments + 1) * (tubularSegments + 1));

    for (uint32_t i = 0; i <= radialSegments; ++i) {
        const float u = (static_cast<float>(i) / static_cast<float>(radialSegments)) * kTwoPi;
        const float cosU = std::cos(u);
        const float sinU = std::sin(u);

        for (uint32_t j = 0; j <= tubularSegments; ++j) {
            const float v = (static_cast<float>(j) / static_cast<float>(tubularSegments)) * kTwoPi;
            const float cosV = std::cos(v);
            const float sinV = std::sin(v);

            Vertex vert{};
            vert.pos.x = (majorRadius + minorRadius * cosV) * cosU;
            vert.pos.y = (majorRadius + minorRadius * cosV) * sinU;
            vert.pos.z = minorRadius * sinV;

            // Outward unit normal from tube ring center
            vert.normal.x = cosV * cosU;
            vert.normal.y = cosV * sinU;
            vert.normal.z = sinV;

            // Task 5: Procedural vertex color
            if (colorMode == 0) {
                // Procedural color based on local coordinate position mapped to [0, 1]
                vert.color.r = std::clamp(0.5f + 0.5f * (vert.pos.x / totalRadius), 0.0f, 1.0f);
                vert.color.g = std::clamp(0.5f + 0.5f * (vert.pos.y / totalRadius), 0.0f, 1.0f);
                vert.color.b = std::clamp(0.5f + 0.5f * (vert.pos.z / minorRadius), 0.0f, 1.0f);
            } else if (colorMode == 1) {
                // Rainbow / Angular mapping based on u and v
                float r = 0.5f + 0.5f * std::sin(u);
                float g = 0.5f + 0.5f * std::sin(u + 2.0944f);
                float b = 0.5f + 0.5f * std::sin(u + 4.1888f);
                float brightness = 0.7f + 0.3f * cosV;
                vert.color = glm::vec3(r, g, b) * brightness;
            } else if (colorMode == 2) {
                // Procedural normal mapping
                vert.color = 0.5f + 0.5f * vert.normal;
            } else {
                // Checkered pattern
                bool check = ((i % 4 < 2) ^ (j % 4 < 2));
                vert.color = check ? glm::vec3(1.0f, 0.9f, 0.7f) : glm::vec3(0.4f, 0.6f, 0.9f);
            }

            mesh.vertices.push_back(vert);
        }
    }

    const uint32_t stride = tubularSegments + 1;
    mesh.indices.reserve(radialSegments * tubularSegments * 6);

    for (uint32_t i = 0; i < radialSegments; ++i) {
        for (uint32_t j = 0; j < tubularSegments; ++j) {
            const uint32_t a = i * stride + j;
            const uint32_t b = (i + 1) * stride + j;
            const uint32_t c = (i + 1) * stride + (j + 1);
            const uint32_t d = i * stride + (j + 1);

            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);

            mesh.indices.push_back(a);
            mesh.indices.push_back(c);
            mesh.indices.push_back(d);
        }
    }

    return mesh;
}

} // namespace graphics
