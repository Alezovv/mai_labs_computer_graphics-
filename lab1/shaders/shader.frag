#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPosition;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 objectColor;
    vec4 lightDir;
    float ambientStrength;
    float specularStrength;
    float shininess;
    float pad;
} ubo;

void main() {
    // Task 5: Multiply procedural vertex color by uniform object color from UI
    vec3 baseColor = fragColor * ubo.objectColor.rgb;

    vec3 N = normalize(fragNormal);
    vec3 L = normalize(ubo.lightDir.xyz);

    // Two-sided lighting for inside/outside of torus
    float diff = max(abs(dot(N, L)), 0.0);
    vec3 diffuse = diff * vec3(1.0);

    vec3 ambient = ubo.ambientStrength * vec3(1.0);

    vec3 finalColor = (ambient + diffuse * (1.0 - ubo.ambientStrength)) * baseColor;
    outColor = vec4(finalColor, ubo.objectColor.a);
}
