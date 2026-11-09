#version 450

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

// Outputs to fragment shader
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;

// Set 0: Scene common data
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

// Set 1: Model specific data
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor; // rgb = spec color, w = shininess
    vec4 params;        // x = ambientMul, y = useVertexColor, zw = pad
} model;

void main() {
    vec4 worldPos = model.model * vec4(inPosition, 1.0);
    gl_Position = scene.proj * scene.view * worldPos;

    outWorldPos = worldPos.xyz;
    outNormal = normalize(mat3(model.normalMatrix) * inNormal);
    outColor = inColor;
}
