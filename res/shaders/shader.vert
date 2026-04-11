#version 450

// Vertex shader — M2
//
// Inputs : 3D positions, uv, normal (giiven by vertex buffer)
// Uniform: UBO binding 0 = MVP matrices 
// Output : clip-space positions + uv + normal world

// Vertex inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

// UBO binding 0 (set 0): MVP matrices (std140 is the layout for UBO in Vulkan)

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Output toward fragment shader
layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormalWorld;  // normale en espace monde
layout(location = 2) out vec3 fragPosWorld;     // position en espace monde (pour eclairage futur)

void main() {
    // Final position in clip space : proj * view * model * pos
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // UV given to the fragment shader
    fragUV = inUV;

    fragNormalWorld = normalize(mat3(transpose(inverse(ubo.model))) * inNormal);

    // world position
    fragPosWorld = vec3(ubo.model * vec4(inPosition, 1.0));
}
