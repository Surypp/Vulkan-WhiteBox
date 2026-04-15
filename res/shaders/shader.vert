#version 450

// Vertex shader — M5
//
// Inputs : 3D positions, uv, normal (given by vertex buffer)
// Uniform: UBO binding 0 = view + proj (stable across draw calls)
//          push constant = model (changes every frame — no descriptor lookup)
// Output : clip-space positions + uv + normal world

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

// push constant: model matrix. written directly into the command buffer via vkCmdPushConstants —
// no descriptor set, no PCIe transfer, no indirection. mat4 = 64 bytes, within the 128-byte min guaranteed by spec.
layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

// view + proj are stable per-frame: kept in the UBO to avoid redundant push constant writes across draw calls.
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormalWorld;
layout(location = 2) out vec3 fragPosWorld;

void main() {
    gl_Position = ubo.proj * ubo.view * pc.model * vec4(inPosition, 1.0);

    fragUV = inUV;

    fragNormalWorld = normalize(mat3(transpose(inverse(pc.model))) * inNormal);

    fragPosWorld = vec3(pc.model * vec4(inPosition, 1.0));
}
