#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace graphics {

class Camera {
public:
    glm::vec3 position{0.0f, 2.5f, 5.0f};
    float yaw = -90.0f;     // Looking along -Z
    float pitch = -20.0f;   // Looking slightly down at origin
    float fov = 60.0f;      // Vertical FOV in degrees
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    // Camera movement settings
    float moveSpeed = 4.0f;
    float mouseSensitivity = 0.15f;

    Camera() = default;

    glm::vec3 getFront() const {
        glm::vec3 front;
        front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        front.y = std::sin(glm::radians(pitch));
        front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        return glm::normalize(front);
    }

    glm::vec3 getRight() const {
        return glm::normalize(glm::cross(getFront(), glm::vec3(0.0f, 1.0f, 0.0f)));
    }

    glm::vec3 getUp() const {
        return glm::normalize(glm::cross(getRight(), getFront()));
    }

    glm::mat4 getViewMatrix() const {
        return glm::lookAt(position, position + getFront(), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 getProjectionMatrix(float aspectRatio) const {
        glm::mat4 proj = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
        // Vulkan clip space has inverted Y compared to OpenGL
        proj[1][1] *= -1.0f;
        return proj;
    }

    void resetToDefault() {
        position = glm::vec3(0.0f, 2.5f, 5.0f);
        yaw = -90.0f;
        pitch = -20.0f;
        fov = 60.0f;
    }

    void lookAtTarget(const glm::vec3& target) {
        glm::vec3 dir = glm::normalize(target - position);
        pitch = glm::degrees(std::asin(glm::clamp(dir.y, -1.0f, 1.0f)));
        yaw = glm::degrees(std::atan2(dir.z, dir.x));
    }
};

} // namespace graphics
