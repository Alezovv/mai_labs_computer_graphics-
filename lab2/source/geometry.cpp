#include "geometry.hpp"

#include <cmath>
#include <algorithm>
#include <glm/gtc/constants.hpp>

namespace graphics {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
} // namespace

Mesh createTorus(float majorRadius, float minorRadius,
                 uint32_t radialSegments, uint32_t tubularSegments,
                 int colorMode) {
    Mesh mesh;
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
            // Position
            vert.pos.x = (majorRadius + minorRadius * cosV) * cosU;
            vert.pos.y = minorRadius * sinV; // Lay torus flat on XZ plane
            vert.pos.z = (majorRadius + minorRadius * cosV) * sinU;

            // True outward normal of torus surface
            vert.normal.x = cosV * cosU;
            vert.normal.y = sinV;
            vert.normal.z = cosV * sinU;
            vert.normal = glm::normalize(vert.normal);

            // Procedural vertex color
            if (colorMode == 0) {
                // Position mapped color
                vert.color.r = std::clamp(0.5f + 0.5f * (vert.pos.x / totalRadius), 0.0f, 1.0f);
                vert.color.g = std::clamp(0.5f + 0.5f * (vert.pos.y / minorRadius), 0.0f, 1.0f);
                vert.color.b = std::clamp(0.5f + 0.5f * (vert.pos.z / totalRadius), 0.0f, 1.0f);
            } else if (colorMode == 1) {
                // Rainbow UV
                float r = 0.5f + 0.5f * std::sin(u);
                float g = 0.5f + 0.5f * std::sin(u + 2.0944f);
                float b = 0.5f + 0.5f * std::sin(u + 4.1888f);
                vert.color = glm::vec3(r, g, b);
            } else {
                // Neutral white
                vert.color = glm::vec3(1.0f);
            }

            mesh.vertices.push_back(vert);
        }
    }

    // Indices (two triangles per quad)
    mesh.indices.reserve(radialSegments * tubularSegments * 6);
    for (uint32_t i = 0; i < radialSegments; ++i) {
        for (uint32_t j = 0; j < tubularSegments; ++j) {
            uint32_t first = i * (tubularSegments + 1) + j;
            uint32_t second = first + tubularSegments + 1;

            mesh.indices.push_back(first);
            mesh.indices.push_back(second);
            mesh.indices.push_back(first + 1);

            mesh.indices.push_back(second);
            mesh.indices.push_back(second + 1);
            mesh.indices.push_back(first + 1);
        }
    }

    return mesh;
}

Mesh createPlane(float width, float depth, uint32_t subdivsX, uint32_t subdivsZ) {
    Mesh mesh;
    subdivsX = std::max(subdivsX, 1u);
    subdivsZ = std::max(subdivsZ, 1u);

    const float halfW = width * 0.5f;
    const float halfD = depth * 0.5f;

    mesh.vertices.reserve((subdivsX + 1) * (subdivsZ + 1));
    for (uint32_t z = 0; z <= subdivsZ; ++z) {
        float fz = static_cast<float>(z) / static_cast<float>(subdivsZ);
        float pz = -halfD + fz * depth;

        for (uint32_t x = 0; x <= subdivsX; ++x) {
            float fx = static_cast<float>(x) / static_cast<float>(subdivsX);
            float px = -halfW + fx * width;

            Vertex vert{};
            vert.pos = glm::vec3(px, 0.0f, pz);
            vert.normal = glm::vec3(0.0f, 1.0f, 0.0f); // Upward normal
            vert.color = glm::vec3(0.9f);
            mesh.vertices.push_back(vert);
        }
    }

    mesh.indices.reserve(subdivsX * subdivsZ * 6);
    for (uint32_t z = 0; z < subdivsZ; ++z) {
        for (uint32_t x = 0; x < subdivsX; ++x) {
            uint32_t row1 = z * (subdivsX + 1);
            uint32_t row2 = (z + 1) * (subdivsX + 1);

            mesh.indices.push_back(row1 + x);
            mesh.indices.push_back(row2 + x);
            mesh.indices.push_back(row1 + x + 1);

            mesh.indices.push_back(row1 + x + 1);
            mesh.indices.push_back(row2 + x);
            mesh.indices.push_back(row2 + x + 1);
        }
    }

    return mesh;
}

Mesh createCube(float size) {
    Mesh mesh;
    const float h = size * 0.5f;

    // 6 faces, 4 vertices per face with proper face normals
    struct FaceDef {
        glm::vec3 n;
        glm::vec3 v0, v1, v2, v3;
    };

    const FaceDef faces[6] = {
        // Front (+Z)
        { { 0.f,  0.f,  1.f}, {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h} },
        // Back (-Z)
        { { 0.f,  0.f, -1.f}, { h, -h, -h}, {-h, -h, -h}, {-h,  h, -h}, { h,  h, -h} },
        // Right (+X)
        { { 1.f,  0.f,  0.f}, { h, -h,  h}, { h, -h, -h}, { h,  h, -h}, { h,  h,  h} },
        // Left (-X)
        { {-1.f,  0.f,  0.f}, {-h, -h, -h}, {-h, -h,  h}, {-h,  h,  h}, {-h,  h, -h} },
        // Top (+Y)
        { { 0.f,  1.f,  0.f}, {-h,  h,  h}, { h,  h,  h}, { h,  h, -h}, {-h,  h, -h} },
        // Bottom (-Y)
        { { 0.f, -1.f,  0.f}, {-h, -h, -h}, { h, -h, -h}, { h, -h,  h}, {-h, -h,  h} }
    };

    for (int i = 0; i < 6; ++i) {
        uint32_t startIndex = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({faces[i].v0, faces[i].n, glm::vec3(1.0f)});
        mesh.vertices.push_back({faces[i].v1, faces[i].n, glm::vec3(1.0f)});
        mesh.vertices.push_back({faces[i].v2, faces[i].n, glm::vec3(1.0f)});
        mesh.vertices.push_back({faces[i].v3, faces[i].n, glm::vec3(1.0f)});

        mesh.indices.push_back(startIndex + 0);
        mesh.indices.push_back(startIndex + 1);
        mesh.indices.push_back(startIndex + 2);
        mesh.indices.push_back(startIndex + 0);
        mesh.indices.push_back(startIndex + 2);
        mesh.indices.push_back(startIndex + 3);
    }

    return mesh;
}

Mesh createConeLines(float length, float angleDegrees, uint32_t segments) {
    Mesh mesh;
    segments = std::max(segments, 8u);

    float angleRad = glm::radians(angleDegrees);
    float radius = length * std::tan(angleRad);

    // Tip at (0, 0, 0), base at (0, 0, -length)
    glm::vec3 tip(0.0f, 0.0f, 0.0f);
    glm::vec3 baseCenter(0.0f, 0.0f, -length);

    // 1. Central direction ray (from tip to base center)
    mesh.vertices.push_back({tip, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f)});
    mesh.vertices.push_back({baseCenter, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f)});
    mesh.indices.push_back(0);
    mesh.indices.push_back(1);

    // 2. Base circle points
    uint32_t circleStart = static_cast<uint32_t>(mesh.vertices.size());
    for (uint32_t i = 0; i < segments; ++i) {
        float theta = (static_cast<float>(i) / static_cast<float>(segments)) * kTwoPi;
        glm::vec3 p(radius * std::cos(theta), radius * std::sin(theta), -length);
        mesh.vertices.push_back({p, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f)});
    }

    // Connect circle vertices as line segments
    for (uint32_t i = 0; i < segments; ++i) {
        mesh.indices.push_back(circleStart + i);
        mesh.indices.push_back(circleStart + ((i + 1) % segments));
    }

    // 3. Side rays from tip to 4 cardinal circle points
    for (uint32_t step = 0; step < 4; ++step) {
        uint32_t circleIdx = circleStart + (step * segments / 4);
        mesh.vertices.push_back({tip, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f)});
        uint32_t tipIdx = static_cast<uint32_t>(mesh.vertices.size() - 1);
        mesh.indices.push_back(tipIdx);
        mesh.indices.push_back(circleIdx);
    }

    return mesh;
}

} // namespace graphics
