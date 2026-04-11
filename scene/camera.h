#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// --- Camera ---
// FPS camera; front/right/up recomputed by UpdateVectors() on each input event.

class Camera {
public:
    glm::vec3 position;
    float     yaw;   // degrees, 0 = looking toward -Z
    float     pitch; // degrees, clamped [-89, 89]

    float moveSpeed        = 2.5f;  // units/second
    float mouseSensitivity = 0.1f;  // degrees per pixel

    float fovDeg = 45.0f;
    float nearZ  = 0.1f;
    float farZ   = 100.0f;

    // derived from yaw/pitch; recomputed by UpdateVectors()
    glm::vec3 front;
    glm::vec3 right;
    glm::vec3 up;

    enum Direction { FORWARD, BACKWARD, LEFT, RIGHT, UP_DIR, DOWN_DIR };

    Camera(glm::vec3 startPos = glm::vec3(0.0f, 0.5f, 3.0f),
        float startYaw = -90.0f,
        float startPitch = -10.0f);

    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix(float aspectRatio) const;

    void ProcessKeyboard(Direction direction, float deltaTime);
    void ProcessMouseMovement(float xOffset, float yOffset);

private:
    void UpdateVectors();
};