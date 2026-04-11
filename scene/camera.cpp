#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

// --- constructor ---

Camera::Camera(glm::vec3 startPos, float startYaw, float startPitch)
    : position(startPos)
    , yaw(startYaw)
    , pitch(startPitch)
{
    UpdateVectors();
}

// --- GetViewMatrix ---

glm::mat4 Camera::GetViewMatrix() const {
    return glm::lookAt(position, position + front, glm::vec3(0.0f, 1.0f, 0.0f));
}

// --- GetProjectionMatrix ---

glm::mat4 Camera::GetProjectionMatrix(float aspectRatio) const {
    glm::mat4 proj = glm::perspective(glm::radians(fovDeg), aspectRatio, nearZ, farZ);
    proj[1][1] *= -1; // GLM targets OpenGL — Vulkan NDC y-axis is inverted
    return proj;
}

// --- ProcessKeyboard ---

void Camera::ProcessKeyboard(Direction direction, float deltaTime) {
    float velocity = moveSpeed * deltaTime;

    // project front onto XZ to prevent WASD from changing altitude
    glm::vec3 flatFront = glm::normalize(glm::vec3(front.x, 0.0f, front.z));

    switch (direction) {
    case FORWARD:  position += flatFront * velocity; break;
    case BACKWARD: position -= flatFront * velocity; break;
    case LEFT:     position -= right * velocity; break;
    case RIGHT:    position += right * velocity; break;
    case UP_DIR:   position += glm::vec3(0.0f, 1.0f, 0.0f) * velocity; break;
    case DOWN_DIR: position -= glm::vec3(0.0f, 1.0f, 0.0f) * velocity; break;
    }
}

// --- ProcessMouseMovement ---

void Camera::ProcessMouseMovement(float xOffset, float yOffset) {
    yaw += xOffset * mouseSensitivity;
    pitch -= yOffset * mouseSensitivity; // inverted: mouse up = positive pitch

    pitch = std::clamp(pitch, -89.0f, 89.0f); // prevent lock at ±90°

    UpdateVectors();
}

// --- UpdateVectors ---
// standard spherical conversion from yaw/pitch to Cartesian front vector

void Camera::UpdateVectors() {
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);

    front = glm::normalize(glm::vec3(
        std::cos(yawRad) * std::cos(pitchRad),
        std::sin(pitchRad),
        std::sin(yawRad) * std::cos(pitchRad)
    ));

    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}