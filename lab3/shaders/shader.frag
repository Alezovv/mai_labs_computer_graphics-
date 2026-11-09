#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec4 outColor;

// ----------------------------------------------------------------------------
// Descriptor Set 0: Scene Uniforms and Light SSBOs
// ----------------------------------------------------------------------------
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

struct PointLight {
    vec4 position;    // xyz = pos, w = intensity
    vec4 color;       // rgb = color
    vec4 attenuation; // x = const, y = lin, z = quad
};

layout(std430, set = 0, binding = 1) readonly buffer PointLightSSBO {
    PointLight pointLights[];
};

struct SpotLight {
    vec4 position;    // xyz = pos, w = intensity
    vec4 direction;   // xyz = normalized direction
    vec4 color;       // rgb = color
    vec4 angles;      // x = innerCutoffCos, y = outerCutoffCos
    vec4 attenuation; // x = const, y = lin, z = quad
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightSSBO {
    SpotLight spotLights[];
};

// ----------------------------------------------------------------------------
// Descriptor Set 1: Dynamic Model UBO & Texture Samplers
// ----------------------------------------------------------------------------
layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor;
    vec4 params; // x = ambientMul, y = useVertexColor, zw = uvScale
} model;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D specularMap;
layout(set = 1, binding = 3) uniform sampler2D emissiveMap;

// ----------------------------------------------------------------------------
// Push Constants: Dynamic non-trivial sampling and runtime controls
// ----------------------------------------------------------------------------
layout(push_constant) uniform PushConstants {
    int nonTrivialSampling;    // 0: standard, 1: waves & multi-tap, 2: dual-flow plasma
    float time;
    float effectSpeed;
    float effectScale;
    vec4 colorModulation;      // Color modulation tint
    float emissiveMultiplier;  // Intensity multiplier for emissive map
} pc;

// Helper: Sample albedo with optional non-trivial dynamic sampling effects (Task 2 for Grade 4)
vec3 sampleAlbedo(vec2 uv) {
    if (pc.nonTrivialSampling == 1) {
        // Effect 1: Wave distortion + multiple sampling of the same image with different UV coordinates
        float t = pc.time * pc.effectSpeed;
        vec2 uv1 = uv + vec2(sin(uv.y * 12.0 + t) * 0.04 * pc.effectScale,
                             cos(uv.x * 12.0 + t) * 0.04 * pc.effectScale);
        vec2 uv2 = uv * 1.6 + vec2(t * 0.08, t * 0.12) * pc.effectScale;

        vec4 col1 = texture(albedoMap, uv1);
        vec4 col2 = texture(albedoMap, uv2);

        // Blend two dynamic samples of the same texture
        float blendFactor = 0.5 + 0.5 * sin(t * 2.0);
        vec3 blended = mix(col1.rgb, col2.rgb, blendFactor);

        // Color modulation
        return blended * pc.colorModulation.rgb;
    }
    else if (pc.nonTrivialSampling == 2) {
        // Effect 2: Dual flow plasma (two opposing velocity fields sampled and blended)
        float t = pc.time * pc.effectSpeed;
        vec2 uvFlowA = uv + vec2(t * 0.15, t * 0.06) * pc.effectScale;
        vec2 uvFlowB = uv - vec2(t * 0.10, -t * 0.14) * pc.effectScale;

        vec3 sampleA = texture(albedoMap, uvFlowA).rgb;
        vec3 sampleB = texture(albedoMap, uvFlowB).rgb;

        vec3 pulse = 0.5 + 0.5 * vec3(sin(t * 2.5), sin(t * 2.5 + 2.0), sin(t * 2.5 + 4.0));
        vec3 mixed = mix(sampleA, sampleB, 0.5) * (pc.colorModulation.rgb + pulse * 0.35);
        return mixed;
    }

    // Default: Standard texture sampling modulated by color tint
    return texture(albedoMap, uv).rgb * pc.colorModulation.rgb;
}

void main() {
    // 1. Albedo texture
    vec3 baseAlbedo = sampleAlbedo(inTexCoord) * model.albedo.rgb;
    if (model.params.y > 0.5) {
        baseAlbedo *= inColor;
    }

    // 2. Specular and Emissive textures (Task 3 for Grade 5)
    vec3 specTex = texture(specularMap, inTexCoord).rgb;
    vec3 emissiveTex = texture(emissiveMap, inTexCoord).rgb;

    vec3 N = normalize(inNormal);
    if (!gl_FrontFacing) {
        N = -N;
    }

    vec3 V = normalize(scene.cameraPos.xyz - inWorldPos);
    float shininess = max(model.specularColor.w, 1.0);
    // Specular map modulates the specular highlight color
    vec3 specColor = model.specularColor.rgb * specTex;

    // 3. Ambient lighting
    vec3 totalAmbient = scene.ambientColor.rgb * baseAlbedo * max(model.params.x, 0.0);
    vec3 totalDiffuse = vec3(0.0);
    vec3 totalSpecular = vec3(0.0);

    // 4. Directional Light (Blinn-Phong)
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

    // 5. Point Lights (Blinn-Phong + Inverse Square Attenuation)
    for (int i = 0; i < scene.numPointLights; ++i) {
        vec3 toLight = pointLights[i].position.xyz - inWorldPos;
        float dist = length(toLight);
        if (dist < 0.0001) continue;

        vec3 L = toLight / dist;
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

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

    // 6. Spotlights (Blinn-Phong + Soft Cone Edges + Attenuation)
    for (int i = 0; i < scene.numSpotLights; ++i) {
        vec3 toLight = spotLights[i].position.xyz - inWorldPos;
        float dist = length(toLight);
        if (dist < 0.0001) continue;

        vec3 L = toLight / dist;
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) continue;

        vec3 spotDir = normalize(spotLights[i].direction.xyz);
        float cosTheta = dot(-L, spotDir);

        float innerCos = spotLights[i].angles.x;
        float outerCos = spotLights[i].angles.y;

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

    // 7. Emissive Glow (Task 3 for Grade 5)
    // Emissive texels add directly to pixel color, bypassing Blinn-Phong calculation
    vec3 totalEmissive = emissiveTex * max(pc.emissiveMultiplier, 0.0);

    vec3 finalColor = totalAmbient + totalDiffuse + totalSpecular + totalEmissive;
    outColor = vec4(finalColor, model.albedo.a);
}
