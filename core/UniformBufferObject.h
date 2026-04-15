#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE   // Vulkan depth range [0,1], not [-1,1]
#include <glm/glm.hpp>

// --- UniformBufferObject ---
// std140: glm::mat4 is natively 16-byte aligned, no alignas() needed.
// one copy per frame in flight to avoid writing a buffer still in use by the GPU.
// model matrix removed — now a push constant (64 bytes, written directly into the CB).
// 192 → 128 bytes per frame slot.

struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 proj;
};