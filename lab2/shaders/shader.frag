#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec4 outColor;

// ----------------------------------------------------------------------------
// Descriptor Set 0: Scene Uniforms and Light Storage Buffers
// ----------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 dirLightDir;   // xyz = normalized direction from light source, w = unused
    vec4 dirLightColor; // rgb = light color, w = intensity
    int numPointLights;
    int numSpotLights;
    int pad1;
    int pad2;
} scene;

struct PointLight {
    vec4 position;    // xyz = position in world, w = intensity
    vec4 color;       // rgb = light color, w = unused
    vec4 attenuation; // x = constant, y = linear, z = quadratic, w = unused
};

layout(std430, set = 0, binding = 1) readonly buffer PointLightSSBO {
    PointLight pointLights[];
};

struct SpotLight {
    vec4 position;    // xyz = position in world, w = intensity
    vec4 direction;   // xyz = normalized beam direction, w = unused
    vec4 color;       // rgb = light color, w = unused
    vec4 angles;      // x = innerCutoffCos, y = outerCutoffCos, zw = unused
    vec4 attenuation; // x = constant, y = linear, z = quadratic, w = unused
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightSSBO {
    SpotLight spotLights[];
};

// ----------------------------------------------------------------------------
// Descriptor Set 1: Model Transforms and Material Properties
// ----------------------------------------------------------------------------
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor; // rgb = specular color, w = shininess exponent
    vec4 params;        // x = ambientMul, y = useVertexColor, zw = pad
} model;

void main() {
    // Determine base albedo (modulate with procedural vertex color if enabled)
    vec3 baseAlbedo = model.albedo.rgb;
    if (model.params.y > 0.5) {
        baseAlbedo *= inColor;
    }

    vec3 N = normalize(inNormal);
    // Double-sided lighting for hollow shapes like torus
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(scene.cameraPos.xyz - inWorldPos);
    float shininess = max(model.specularColor.w, 1.0);
    vec3 specColor = model.specularColor.rgb;

    // 1. Ambient lighting
    vec3 totalAmbient = scene.ambientColor.rgb * baseAlbedo * max(model.params.x, 0.0);
    vec3 totalDiffuse = vec3(0.0);
    vec3 totalSpecular = vec3(0.0);

    // 2. Directional light (Blinn-Phong)
    if (scene.dirLightColor.w > 0.0) {
        vec3 L = normalize(-scene.dirLightDir.xyz);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            vec3 H = normalize(L + V);
            float NdotH = max(dot(N, H), 0.0);
            float spec = pow(NdotH, shininess);

            vec3 lightRadiance = scene.dirLightColor.rgb * scene.dirLightColor.w;
            totalDiffuse += NdotL * lightRadiance * baseAlbedo;
            totalSpecular += spec * lightRadiance * specColor;
        }
    }

    // 3. Point lights (Blinn-Phong + Inverse Square Attenuation from SSBO)
    for (int i = 0; i < scene.numPointLights; ++i) {
        vec3 toLight = pointLights[i].position.xyz - inWorldPos;
        float dist = length(toLight);
        if (dist < 0.0001) continue;

        vec3 L = toLight / dist;
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        // Inverse-square law attenuation: 1 / (kc + kl*d + kq*d^2)
        float att = 1.0 / (pointLights[i].attenuation.x +
                           pointLights[i].attenuation.y * dist +
                           pointLights[i].attenuation.z * dist * dist);
        float intensity = pointLights[i].position.w * att;
        if (intensity <= 0.0) continue;

        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float spec = pow(NdotH, shininess);

        vec3 lightRadiance = pointLights[i].color.rgb * intensity;
        totalDiffuse += NdotL * lightRadiance * baseAlbedo;
        totalSpecular += spec * lightRadiance * specColor;
    }

    // 4. Spotlights (Blinn-Phong + Soft Edges + Inverse Square Attenuation from SSBO)
    for (int i = 0; i < scene.numSpotLights; ++i) {
        vec3 toLight = spotLights[i].position.xyz - inWorldPos;
        float dist = length(toLight);
        if (dist < 0.0001) continue;

        vec3 L = toLight / dist;
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        // Angle between light direction vector and ray to the surface
        vec3 spotDir = normalize(spotLights[i].direction.xyz);
        float cosTheta = dot(-L, spotDir);

        float innerCos = spotLights[i].angles.x;
        float outerCos = spotLights[i].angles.y;

        // Soft spotlight boundary calculation:
        // intensity scales smoothly from 1.0 at inner angle to 0.0 at outer angle
        float epsilon = innerCos - outerCos;
        float spotFactor = clamp((cosTheta - outerCos) / max(epsilon, 0.00001), 0.0, 1.0);
        if (spotFactor <= 0.0) continue;

        float att = 1.0 / (spotLights[i].attenuation.x +
                           spotLights[i].attenuation.y * dist +
                           spotLights[i].attenuation.z * dist * dist);
        float intensity = spotLights[i].position.w * att * spotFactor;
        if (intensity <= 0.0) continue;

        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float spec = pow(NdotH, shininess);

        vec3 lightRadiance = spotLights[i].color.rgb * intensity;
        totalDiffuse += NdotL * lightRadiance * baseAlbedo;
        totalSpecular += spec * lightRadiance * specColor;
    }

    vec3 finalColor = totalAmbient + totalDiffuse + totalSpecular;
    outColor = vec4(finalColor, model.albedo.a);
}
