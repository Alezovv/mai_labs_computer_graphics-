#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 dirLightDir;
    vec4 dirLightColor;
    int numPointLights;
    int numSpotLights;
    int pad1;
    int pad2;
} scene;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} pc;

layout(location = 0) out vec4 fragColor;

void main() {
    gl_Position = scene.proj * scene.view * pc.model * vec4(inPosition, 1.0);
    fragColor = pc.color;
}
