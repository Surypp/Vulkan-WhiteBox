#version 450

// Fragment shader — M2
//
// /Displays the cube with a simple directional lightning
// (Phong diffus + ambiant) 

// Input from vertex shader
layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormalWorld;
layout(location = 2) in vec3 fragPosWorld;

// Texture 
layout(binding = 1) uniform sampler2D texSampler;

// Color output
layout(location = 0) out vec4 outColor;

void main() {
    vec3 baseColor = texture(texSampler, fragUV).rgb;

    // fixed directional lightning (from top left in front)
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));

    float ambient = 0.25;
    float diffuse = max(dot(fragNormalWorld, lightDir), 0.0);

    // final color (ambiant + diffuse)
    vec3 finalColor = baseColor * (ambient + diffuse);

    outColor = vec4(finalColor, 1.0);
}
