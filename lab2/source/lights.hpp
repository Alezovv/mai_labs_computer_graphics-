#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cmath>

namespace graphics {

// GPU memory layout structures matching GLSL std430
struct alignas(16) PointLightGPU {
    glm::vec4 position{0.0f, 2.0f, 0.0f, 1.0f};      // xyz = pos, w = intensity
    glm::vec4 color{1.0f, 0.6f, 0.2f, 0.0f};         // rgb = color
    glm::vec4 attenuation{1.0f, 0.09f, 0.032f, 0.0f}; // x = const, y = lin, z = quad
};

struct alignas(16) SpotLightGPU {
    glm::vec4 position{0.0f, 3.5f, 2.0f, 2.0f};        // xyz = pos, w = intensity
    glm::vec4 direction{0.0f, -0.85f, -0.52f, 0.0f};   // xyz = normalized direction
    glm::vec4 color{0.2f, 0.7f, 1.0f, 0.0f};           // rgb = color
    glm::vec4 angles{0.9659f, 0.9063f, 0.0f, 0.0f};    // x = cos(inner), y = cos(outer)
    glm::vec4 attenuation{1.0f, 0.07f, 0.017f, 0.0f}; // attenuation factors
};

// UI and control wrappers
struct PointLight {
    std::string name;
    glm::vec3 position{0.0f, 2.0f, 0.0f};
    glm::vec3 color{1.0f, 0.6f, 0.2f};
    float intensity = 1.5f;

    // Attenuation coefficients: 1 / (kc + kl*d + kq*d^2)
    float constantAtt = 1.0f;
    float linearAtt = 0.09f;
    float quadraticAtt = 0.032f;

    bool enabled = true;
    bool animate = false;
    float animRadius = 2.5f;
    float animSpeed = 1.0f;
    float animHeight = 1.5f;
    float animPhase = 0.0f;

    PointLightGPU toGPU(float time) const {
        PointLightGPU gpu{};
        glm::vec3 pos = position;
        if (animate) {
            float angle = time * animSpeed + animPhase;
            pos.x = std::cos(angle) * animRadius;
            pos.z = std::sin(angle) * animRadius;
            pos.y = animHeight + 0.3f * std::sin(angle * 2.0f);
        }
        gpu.position = glm::vec4(pos, enabled ? intensity : 0.0f);
        gpu.color = glm::vec4(color, 0.0f);
        gpu.attenuation = glm::vec4(constantAtt, linearAtt, quadraticAtt, 0.0f);
        return gpu;
    }

    glm::vec3 getActivePosition(float time) const {
        if (!animate) return position;
        float angle = time * animSpeed + animPhase;
        return glm::vec3(std::cos(angle) * animRadius,
                         animHeight + 0.3f * std::sin(angle * 2.0f),
                         std::sin(angle) * animRadius);
    }
};

struct SpotLight {
    std::string name;
    glm::vec3 position{0.0f, 3.5f, 2.0f};
    glm::vec3 direction{0.0f, -0.85f, -0.52f};
    glm::vec3 color{0.2f, 0.7f, 1.0f};
    float intensity = 2.5f;

    // Angles in degrees
    float innerAngleDegrees = 15.0f;
    float outerAngleDegrees = 25.0f;

    // Attenuation coefficients
    float constantAtt = 1.0f;
    float linearAtt = 0.07f;
    float quadraticAtt = 0.017f;

    bool enabled = true;
    bool animate = false;
    float animSpeed = 0.8f;
    float animRadius = 3.0f;

    SpotLightGPU toGPU(float time) const {
        SpotLightGPU gpu{};
        glm::vec3 pos = position;
        glm::vec3 dir = glm::normalize(direction);

        if (animate) {
            float angle = time * animSpeed;
            pos.x = std::cos(angle) * animRadius;
            pos.z = std::sin(angle) * animRadius;
            // Point towards center / torus
            dir = glm::normalize(glm::vec3(0.0f, 0.5f, 0.0f) - pos);
        }

        gpu.position = glm::vec4(pos, enabled ? intensity : 0.0f);
        gpu.direction = glm::vec4(dir, 0.0f);
        gpu.color = glm::vec4(color, 0.0f);

        // Convert angles to cosines
        float inRad = glm::radians(innerAngleDegrees);
        float outRad = glm::radians(std::max(outerAngleDegrees, innerAngleDegrees));
        gpu.angles = glm::vec4(std::cos(inRad), std::cos(outRad), 0.0f, 0.0f);
        gpu.attenuation = glm::vec4(constantAtt, linearAtt, quadraticAtt, 0.0f);
        return gpu;
    }

    glm::vec3 getActivePosition(float time) const {
        if (!animate) return position;
        float angle = time * animSpeed;
        return glm::vec3(std::cos(angle) * animRadius, position.y, std::sin(angle) * animRadius);
    }

    glm::vec3 getActiveDirection(float time) const {
        if (!animate) return glm::normalize(direction);
        glm::vec3 pos = getActivePosition(time);
        return glm::normalize(glm::vec3(0.0f, 0.5f, 0.0f) - pos);
    }
};

} // namespace graphics
