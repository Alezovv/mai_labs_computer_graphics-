#include "application.hpp"
#include "camera.hpp"
#include "lights.hpp"
#include "geometry.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

namespace application {

const char* getTitle() {
    return "MAI Computer Graphics - Lab 2: Camera & Blinn-Phong Lighting";
}

namespace {

// ----------------------------------------------------------------------------
// GPU Uniform & Push Constant Structures
// ----------------------------------------------------------------------------
struct alignas(16) SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 cameraPos;
    glm::vec4 ambientColor;
    glm::vec4 dirLightDir;   // xyz = normalized dir from light source, w = unused
    glm::vec4 dirLightColor; // rgb = color, w = intensity
    int32_t numPointLights = 0;
    int32_t numSpotLights = 0;
    int32_t pad1 = 0;
    int32_t pad2 = 0;
};

struct alignas(16) ModelUBO {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    glm::vec4 albedo;
    glm::vec4 specularColor; // rgb = spec color, w = shininess
    glm::vec4 params;        // x = ambientMul, y = useVertexColor, zw = pad
};

struct PushConstantBlock {
    glm::mat4 model;
    glm::vec4 color;
};

// ----------------------------------------------------------------------------
// Vulkan Buffer Resource
// ----------------------------------------------------------------------------
struct BufferResource {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VmaAllocationCreateFlags allocFlags,
                  BufferResource& outBuffer) {
    auto& ctx = graphics::internal::context;

    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VmaAllocationCreateInfo allocInfo = {
        .flags = allocFlags,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    if (allocFlags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) {
        allocInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VmaAllocationInfo resultAllocInfo{};
    VkResult res = vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo,
                                   &outBuffer.buffer, &outBuffer.allocation,
                                   &resultAllocInfo);
    if (res != VK_SUCCESS) {
        std::cerr << "Failed to allocate Vulkan buffer of size " << size << '\n';
        return false;
    }

    outBuffer.size = size;
    outBuffer.mapped = resultAllocInfo.pMappedData;
    return true;
}

void destroyBuffer(BufferResource& buf) {
    auto& ctx = graphics::internal::context;
    if (buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, buf.buffer, buf.allocation);
        buf.buffer = VK_NULL_HANDLE;
        buf.allocation = VK_NULL_HANDLE;
        buf.mapped = nullptr;
        buf.size = 0;
    }
}

std::vector<char> readFile(const std::string& filename) {
    const std::vector<std::string> candidates = {
        filename,
        "lab2/" + filename,
        "../" + filename,
        "../lab2/" + filename,
        "../../" + filename,
        "../../lab2/" + filename,
        "shaders/" + filename,
        "lab2/shaders/" + filename,
    };
    for (const auto& path : candidates) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            size_t fileSize = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            return buffer;
        }
    }
    std::cerr << "Could not open shader file: " << filename << '\n';
    return {};
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    if (code.empty()) {
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module\n";
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

// ----------------------------------------------------------------------------
// Scene Object Definition
// ----------------------------------------------------------------------------
struct SceneObject {
    std::string name;
    bool visible = true;

    // GPU buffers
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    uint32_t indexCount = 0;

    // Model Uniform Buffer & Descriptor Set (Set 1)
    BufferResource modelBuffer;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    // Transform
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 rotation{0.0f, 0.0f, 0.0f}; // Euler degrees
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    // Material parameters (Blinn-Phong)
    glm::vec4 albedo{0.9f, 0.7f, 0.2f, 1.0f};
    glm::vec3 specularColor{1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
    float ambientMultiplier = 1.0f;
    bool useVertexColor = false;

    glm::mat4 computeModelMatrix() const {
        glm::mat4 m(1.0f);
        m = glm::translate(m, position);
        m = glm::rotate(m, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        m = glm::rotate(m, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        m = glm::rotate(m, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        m = glm::scale(m, scale);
        return m;
    }

    glm::mat4 computeNormalMatrix() const {
        return glm::transpose(glm::inverse(computeModelMatrix()));
    }

    void updateUBO() {
        if (!modelBuffer.mapped) return;
        ModelUBO ubo{};
        ubo.model = computeModelMatrix();
        ubo.normalMatrix = computeNormalMatrix();
        ubo.albedo = albedo;
        ubo.specularColor = glm::vec4(specularColor, shininess);
        ubo.params = glm::vec4(ambientMultiplier, useVertexColor ? 1.0f : 0.0f, 0.0f, 0.0f);
        std::memcpy(modelBuffer.mapped, &ubo, sizeof(ubo));
    }
};

// ----------------------------------------------------------------------------
// Global Application State
// ----------------------------------------------------------------------------
graphics::Camera camera;

// Global Lighting State
glm::vec3 ambientLightColor{0.15f, 0.15f, 0.18f};
float ambientIntensity = 1.0f;

// Directional Light
bool dirLightEnabled = true;
glm::vec3 dirLightDirection{0.5f, 1.0f, 0.3f};
glm::vec3 dirLightColor{1.0f, 0.95f, 0.85f};
float dirLightIntensity = 1.0f;

// Point Lights & Spotlights
std::vector<graphics::PointLight> pointLights;
std::vector<graphics::SpotLight> spotLights;

constexpr uint32_t kMaxLights = 64;
BufferResource pointLightSSBO;
BufferResource spotLightSSBO;

// Scene UBO (Set 0)
BufferResource sceneUboBuffer;

// Vulkan Descriptors & Pipelines
VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;

VkPipelineLayout mainPipelineLayout = VK_NULL_HANDLE;
VkPipeline mainPipeline = VK_NULL_HANDLE;

// Light Visualizer Pipelines (Task 4 for Grade 5)
VkPipelineLayout unlitPipelineLayout = VK_NULL_HANDLE;
VkPipeline unlitMarkerPipeline = VK_NULL_HANDLE;
VkPipeline unlitConePipeline = VK_NULL_HANDLE;

// Meshes for visualizers
BufferResource cubeVertexBuffer;
BufferResource cubeIndexBuffer;
uint32_t cubeIndexCount = 0;

BufferResource coneVertexBuffer;
BufferResource coneIndexBuffer;
uint32_t coneIndexCount = 0;

// Scene Objects
SceneObject torusObject;
SceneObject floorObject;

// Torus settings
float torusMajorRadius = 1.5f;
float torusMinorRadius = 0.5f;
int torusRadialSegments = 48;
int torusTubularSegments = 24;
int torusColorMode = 0;
bool torusMeshDirty = false;

// Flags & Settings
bool showLightVisualizers = true;
bool animateLights = true;
float currentTime = 0.0f;

// ----------------------------------------------------------------------------
// Mesh Upload Helper
// ----------------------------------------------------------------------------
bool uploadMesh(const graphics::Mesh& mesh, BufferResource& vb, BufferResource& ib, uint32_t& outIndexCount) {
    destroyBuffer(vb);
    destroyBuffer(ib);

    VkDeviceSize vSize = sizeof(graphics::Vertex) * mesh.vertices.size();
    VkDeviceSize iSize = sizeof(uint32_t) * mesh.indices.size();

    if (!createBuffer(vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, vb)) {
        return false;
    }
    std::memcpy(vb.mapped, mesh.vertices.data(), vSize);

    if (!createBuffer(iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, ib)) {
        return false;
    }
    std::memcpy(ib.mapped, mesh.indices.data(), iSize);

    outIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return true;
}

void rebuildTorusMesh() {
    auto mesh = graphics::createTorus(torusMajorRadius, torusMinorRadius,
                                      torusRadialSegments, torusTubularSegments,
                                      torusColorMode);
    uploadMesh(mesh, torusObject.vertexBuffer, torusObject.indexBuffer, torusObject.indexCount);
    torusMeshDirty = false;
}

// ----------------------------------------------------------------------------
// Descriptor Pool & Sets Initialization
// ----------------------------------------------------------------------------
bool initDescriptors() {
    auto& ctx = graphics::internal::context;

    // 1. Scene Descriptor Set Layout (Set 0)
    // Binding 0: Scene UBO
    // Binding 1: PointLight SSBO
    // Binding 2: SpotLight SSBO
    VkDescriptorSetLayoutBinding sceneBindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        }
    };

    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = sceneBindings,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr, &sceneSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create Scene descriptor set layout\n";
        return false;
    }

    // 2. Model Descriptor Set Layout (Set 1)
    // Binding 0: Model UBO
    VkDescriptorSetLayoutBinding modelBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    VkDescriptorSetLayoutCreateInfo modelLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &modelBinding,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr, &modelSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create Model descriptor set layout\n";
        return false;
    }

    // 3. Descriptor Pool
    VkDescriptorPoolSize poolSizes[] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 16 },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8 },
    };

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 16,
        .poolSizeCount = 2,
        .pPoolSizes = poolSizes,
    };
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    // 4. Allocate and Update Scene Descriptor Set (Set 0)
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &sceneSetLayout,
    };
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &sceneDescriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate Scene descriptor set\n";
        return false;
    }

    VkDescriptorBufferInfo sceneUboInfo = {
        .buffer = sceneUboBuffer.buffer,
        .offset = 0,
        .range = sizeof(SceneUBO),
    };
    VkDescriptorBufferInfo pointSsboInfo = {
        .buffer = pointLightSSBO.buffer,
        .offset = 0,
        .range = pointLightSSBO.size,
    };
    VkDescriptorBufferInfo spotSsboInfo = {
        .buffer = spotLightSSBO.buffer,
        .offset = 0,
        .range = spotLightSSBO.size,
    };

    VkWriteDescriptorSet descriptorWrites[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneDescriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &sceneUboInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneDescriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &pointSsboInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneDescriptorSet,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &spotSsboInfo,
        }
    };
    vkUpdateDescriptorSets(ctx.device, 3, descriptorWrites, 0, nullptr);

    return true;
}

bool createModelDescriptorSet(SceneObject& obj) {
    auto& ctx = graphics::internal::context;

    // Create UBO for model
    if (!createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      obj.modelBuffer)) {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &modelSetLayout,
    };
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &obj.descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate model descriptor set\n";
        return false;
    }

    VkDescriptorBufferInfo modelInfo = {
        .buffer = obj.modelBuffer.buffer,
        .offset = 0,
        .range = sizeof(ModelUBO),
    };

    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = obj.descriptorSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &modelInfo,
    };
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return true;
}

// ----------------------------------------------------------------------------
// Graphics Pipelines Creation
// ----------------------------------------------------------------------------
bool initPipelines() {
    auto& ctx = graphics::internal::context;

    // 1. Main Blinn-Phong Pipeline
    VkDescriptorSetLayout setLayouts[] = { sceneSetLayout, modelSetLayout };
    VkPipelineLayoutCreateInfo mainLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = setLayouts,
    };
    if (vkCreatePipelineLayout(ctx.device, &mainLayoutInfo, nullptr, &mainPipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create main pipeline layout\n";
        return false;
    }

    auto vertCode = readFile("shaders/shader.vert.spv");
    auto fragCode = readFile("shaders/shader.frag.spv");
    if (vertCode.empty() || fragCode.empty()) {
        std::cerr << "Failed to read main shader binaries\n";
        return false;
    }

    VkShaderModule vertModule = createShaderModule(ctx.device, vertCode);
    VkShaderModule fragModule = createShaderModule(ctx.device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertModule,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName = "main",
        }
    };

    auto bindingDesc = graphics::Vertex::getBindingDescription();
    auto attrDescs = graphics::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDesc,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size()),
        .pVertexAttributeDescriptions = attrDescs.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE, // Two-sided for torus
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = mainPipelineLayout,
        .renderPass = ctx.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mainPipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create main graphics pipeline\n";
        return false;
    }

    vkDestroyShaderModule(ctx.device, vertModule, nullptr);
    vkDestroyShaderModule(ctx.device, fragModule, nullptr);

    // 2. Unlit Visualizer Pipeline (Markers & Cones for Task 4)
    VkPushConstantRange pushConstantRange = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushConstantBlock),
    };

    VkPipelineLayoutCreateInfo unlitLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &sceneSetLayout, // Reuses SceneUBO for view & proj!
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    if (vkCreatePipelineLayout(ctx.device, &unlitLayoutInfo, nullptr, &unlitPipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create unlit pipeline layout\n";
        return false;
    }

    auto unlitVertCode = readFile("shaders/unlit.vert.spv");
    auto unlitFragCode = readFile("shaders/unlit.frag.spv");
    if (!unlitVertCode.empty() && !unlitFragCode.empty()) {
        VkShaderModule uVertModule = createShaderModule(ctx.device, unlitVertCode);
        VkShaderModule uFragModule = createShaderModule(ctx.device, unlitFragCode);

        VkPipelineShaderStageCreateInfo unlitStages[] = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = uVertModule,
                .pName = "main",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = uFragModule,
                .pName = "main",
            }
        };

        // Unlit Triangles (Cubes for lights)
        pipelineInfo.pStages = unlitStages;
        pipelineInfo.layout = unlitPipelineLayout;
        pipelineInfo.pInputAssemblyState = &inputAssembly; // TRIANGLE_LIST
        depthStencil.depthWriteEnable = VK_TRUE;
        pipelineInfo.pDepthStencilState = &depthStencil;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &unlitMarkerPipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create unlit marker pipeline\n";
        }

        // Unlit Lines (Cone lines for spotlights)
        VkPipelineInputAssemblyStateCreateInfo lineAssembly = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        };
        pipelineInfo.pInputAssemblyState = &lineAssembly;
        depthStencil.depthWriteEnable = VK_FALSE; // Transparent lines feel
        pipelineInfo.pDepthStencilState = &depthStencil;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &unlitConePipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create unlit cone lines pipeline\n";
        }

        vkDestroyShaderModule(ctx.device, uVertModule, nullptr);
        vkDestroyShaderModule(ctx.device, uFragModule, nullptr);
    }

    return true;
}

} // namespace

// ----------------------------------------------------------------------------
// Public Application Interface
// ----------------------------------------------------------------------------
bool initialize() {
    // 1. Allocate Scene Uniform Buffer
    if (!createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      sceneUboBuffer)) {
        return false;
    }

    // 2. Allocate Storage Buffers for Point & Spot lights
    VkDeviceSize pointSsboSize = sizeof(graphics::PointLightGPU) * kMaxLights;
    if (!createBuffer(pointSsboSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      pointLightSSBO)) {
        return false;
    }

    VkDeviceSize spotSsboSize = sizeof(graphics::SpotLightGPU) * kMaxLights;
    if (!createBuffer(spotSsboSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      spotLightSSBO)) {
        return false;
    }

    // 3. Setup Descriptors & Pipelines
    if (!initDescriptors()) return false;
    if (!initPipelines()) return false;

    // 4. Create Main Torus (Variant 7)
    torusObject.name = "Тор (Вариант 7)";
    torusObject.position = glm::vec3(0.0f, 1.0f, 0.0f);
    torusObject.albedo = glm::vec4(1.0f, 0.75f, 0.2f, 1.0f);
    torusObject.specularColor = glm::vec3(1.0f, 1.0f, 1.0f);
    torusObject.shininess = 64.0f;
    torusObject.useVertexColor = true;
    createModelDescriptorSet(torusObject);
    rebuildTorusMesh();

    // 5. Create Floor Plane
    floorObject.name = "Пол (Сетка)";
    floorObject.position = glm::vec3(0.0f, 0.0f, 0.0f);
    floorObject.albedo = glm::vec4(0.35f, 0.38f, 0.45f, 1.0f);
    floorObject.specularColor = glm::vec3(0.4f, 0.4f, 0.4f);
    floorObject.shininess = 16.0f;
    floorObject.useVertexColor = false;
    createModelDescriptorSet(floorObject);
    auto floorMesh = graphics::createPlane(20.0f, 20.0f, 40, 40);
    uploadMesh(floorMesh, floorObject.vertexBuffer, floorObject.indexBuffer, floorObject.indexCount);

    // 6. Create Visualizer Meshes (Cube & Cone Lines)
    auto cubeMesh = graphics::createCube(0.2f);
    uploadMesh(cubeMesh, cubeVertexBuffer, cubeIndexBuffer, cubeIndexCount);

    auto coneMesh = graphics::createConeLines(2.5f, 25.0f, 32);
    uploadMesh(coneMesh, coneVertexBuffer, coneIndexBuffer, coneIndexCount);

    // 7. Add Initial Default Lights
    // Default Point Light 1 (Warm orange)
    {
        graphics::PointLight p1;
        p1.name = "Точечный #1 (Оранжевый)";
        p1.position = glm::vec3(2.5f, 1.8f, 0.0f);
        p1.color = glm::vec3(1.0f, 0.5f, 0.1f);
        p1.intensity = 2.0f;
        p1.animate = true;
        p1.animRadius = 3.0f;
        p1.animSpeed = 1.0f;
        p1.animPhase = 0.0f;
        pointLights.push_back(p1);
    }
    // Default Point Light 2 (Cyan)
    {
        graphics::PointLight p2;
        p2.name = "Точечный #2 (Бирюзовый)";
        p2.position = glm::vec3(-2.5f, 1.8f, 0.0f);
        p2.color = glm::vec3(0.1f, 0.8f, 1.0f);
        p2.intensity = 2.0f;
        p2.animate = true;
        p2.animRadius = 3.0f;
        p2.animSpeed = 1.0f;
        p2.animPhase = 3.14159f; // opposite phase
        pointLights.push_back(p2);
    }

    // Default Spotlight 1 (Cold Blue)
    {
        graphics::SpotLight s1;
        s1.name = "Прожектор #1 (Синий)";
        s1.position = glm::vec3(0.0f, 4.0f, 2.5f);
        s1.direction = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f) - s1.position);
        s1.color = glm::vec3(0.4f, 0.7f, 1.0f);
        s1.intensity = 3.5f;
        s1.innerAngleDegrees = 15.0f;
        s1.outerAngleDegrees = 25.0f;
        s1.animate = false;
        spotLights.push_back(s1);
    }

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;

    vkDeviceWaitIdle(ctx.device);

    destroyBuffer(cubeVertexBuffer);
    destroyBuffer(cubeIndexBuffer);
    destroyBuffer(coneVertexBuffer);
    destroyBuffer(coneIndexBuffer);

    destroyBuffer(torusObject.vertexBuffer);
    destroyBuffer(torusObject.indexBuffer);
    destroyBuffer(torusObject.modelBuffer);

    destroyBuffer(floorObject.vertexBuffer);
    destroyBuffer(floorObject.indexBuffer);
    destroyBuffer(floorObject.modelBuffer);

    destroyBuffer(sceneUboBuffer);
    destroyBuffer(pointLightSSBO);
    destroyBuffer(spotLightSSBO);

    if (mainPipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, mainPipeline, nullptr);
    if (mainPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, mainPipelineLayout, nullptr);

    if (unlitMarkerPipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, unlitMarkerPipeline, nullptr);
    if (unlitConePipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, unlitConePipeline, nullptr);
    if (unlitPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, unlitPipelineLayout, nullptr);

    if (descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    if (sceneSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout, nullptr);
    if (modelSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, modelSetLayout, nullptr);
}

void update(double time) {
    currentTime = static_cast<float>(time);

    if (torusMeshDirty) {
        rebuildTorusMesh();
    }

    // ------------------------------------------------------------------------
    // Mouse Navigation for Camera (Hold Right Mouse Button to rotate & scroll zoom)
    // ------------------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            camera.yaw += io.MouseDelta.x * camera.mouseSensitivity;
            camera.pitch -= io.MouseDelta.y * camera.mouseSensitivity;
            camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
        }
        if (io.MouseWheel != 0.0f) {
            camera.fov = std::clamp(camera.fov - io.MouseWheel * 2.0f, 15.0f, 110.0f);
        }
    }

    // WASD movement when right mouse button is pressed
    if (!io.WantCaptureKeyboard && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        float speed = camera.moveSpeed * io.DeltaTime;
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) speed *= 2.5f;

        if (ImGui::IsKeyDown(ImGuiKey_W)) camera.position += camera.getFront() * speed;
        if (ImGui::IsKeyDown(ImGuiKey_S)) camera.position -= camera.getFront() * speed;
        if (ImGui::IsKeyDown(ImGuiKey_A)) camera.position -= camera.getRight() * speed;
        if (ImGui::IsKeyDown(ImGuiKey_D)) camera.position += camera.getRight() * speed;
        if (ImGui::IsKeyDown(ImGuiKey_E) || ImGui::IsKeyDown(ImGuiKey_Space)) camera.position += glm::vec3(0, 1, 0) * speed;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) camera.position -= glm::vec3(0, 1, 0) * speed;
    }

    // ------------------------------------------------------------------------
    // ImGUI User Interface
    // ------------------------------------------------------------------------
    // Window 1: Camera & Scene Settings
    ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 520), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Камера и Сцена")) {
        if (ImGui::CollapsingHeader("Параметры Камеры", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Позиция", glm::value_ptr(camera.position), 0.05f);
            ImGui::SliderFloat("Поворот (Yaw)", &camera.yaw, -180.0f, 180.0f, "%.1f°");
            ImGui::SliderFloat("Наклон (Pitch)", &camera.pitch, -89.0f, 89.0f, "%.1f°");
            ImGui::SliderFloat("Угол обзора (FOV)", &camera.fov, 20.0f, 110.0f, "%.1f°");

            ImGui::TextDisabled("Пресеты вида:");
            if (ImGui::Button("Спереди")) {
                camera.position = glm::vec3(0.0f, 1.5f, 5.0f);
                camera.yaw = -90.0f; camera.pitch = -5.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Сверху")) {
                camera.position = glm::vec3(0.0f, 7.0f, 0.01f);
                camera.yaw = -90.0f; camera.pitch = -89.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Изометрия")) {
                camera.position = glm::vec3(4.0f, 3.5f, 4.0f);
                camera.lookAtTarget(glm::vec3(0.0f, 1.0f, 0.0f));
            }
            ImGui::SameLine();
            if (ImGui::Button("Сброс")) {
                camera.resetToDefault();
            }

            ImGui::Separator();
            ImGui::TextWrapped("Управление 3D: зажмите [ПКМ] для вращения обзора, клавиши [W, A, S, D, Q, E] для полета.");
        }

        if (ImGui::CollapsingHeader("Рассеянный свет (Ambient)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Цвет рассеянного света", glm::value_ptr(ambientLightColor));
            ImGui::SliderFloat("Интенсивность рассеяния", &ambientIntensity, 0.0f, 2.0f);
        }

        if (ImGui::CollapsingHeader("Опции отображения (Оценка 5)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Показывать визуализаторы источников", &showLightVisualizers);
            ImGui::Checkbox("Анимация вращения источников", &animateLights);
        }
    }
    ImGui::End();

    // Window 2: Light Sources (Directional, Point, Spot)
    ImGui::SetNextWindowPos(ImVec2(410, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 680), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Источники освещения")) {
        // --- 1. Directional Light ---
        if (ImGui::CollapsingHeader("1. Направленный свет (Directional)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Включен##Dir", &dirLightEnabled);
            ImGui::SliderFloat3("Направление", glm::value_ptr(dirLightDirection), -1.0f, 1.0f);
            ImGui::ColorEdit3("Цвет##Dir", glm::value_ptr(dirLightColor));
            ImGui::SliderFloat("Интенсивность##Dir", &dirLightIntensity, 0.0f, 5.0f);
        }

        // --- 2. Point Lights (SSBO) ---
        if (ImGui::CollapsingHeader("2. Точечные источники (SSBO)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Активно точечных источников: %zu / %u", pointLights.size(), kMaxLights);
            if (pointLights.size() < kMaxLights) {
                if (ImGui::Button("+ Добавить точечный источник")) {
                    graphics::PointLight p;
                    p.name = "Точечный #" + std::to_string(pointLights.size() + 1);
                    p.position = glm::vec3(0.0f, 2.0f, 1.5f);
                    p.color = glm::vec3(1.0f, 1.0f, 0.5f);
                    p.intensity = 2.0f;
                    p.animate = false;
                    pointLights.push_back(p);
                }
            }

            for (size_t i = 0; i < pointLights.size(); ++i) {
                auto& p = pointLights[i];
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode(p.name.c_str())) {
                    ImGui::Checkbox("Активен", &p.enabled);
                    ImGui::SameLine();
                    if (ImGui::Button("Удалить источник")) {
                        pointLights.erase(pointLights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }

                    ImGui::DragFloat3("Позиция", glm::value_ptr(p.position), 0.05f);
                    ImGui::ColorEdit3("Цвет", glm::value_ptr(p.color));
                    ImGui::SliderFloat("Интенсивность", &p.intensity, 0.0f, 10.0f);

                    ImGui::Checkbox("Орбитальная анимация", &p.animate);
                    if (p.animate) {
                        ImGui::SliderFloat("Радиус орбиты", &p.animRadius, 0.5f, 6.0f);
                        ImGui::SliderFloat("Скорость орбиты", &p.animSpeed, 0.1f, 3.0f);
                        ImGui::SliderFloat("Высота", &p.animHeight, 0.2f, 4.0f);
                    }

                    ImGui::TextDisabled("Затухание 1 / (kc + kl*d + kq*d^2):");
                    ImGui::DragFloat("Константа (kc)", &p.constantAtt, 0.01f, 0.1f, 2.0f);
                    ImGui::DragFloat("Линейный (kl)", &p.linearAtt, 0.005f, 0.0f, 1.0f, "%.4f");
                    ImGui::DragFloat("Квадратичный (kq)", &p.quadraticAtt, 0.001f, 0.0f, 0.5f, "%.5f");

                    ImGui::TextDisabled("Пресеты дистанции:");
                    if (ImGui::Button("7 метров"))  { p.constantAtt = 1.0f; p.linearAtt = 0.7f;   p.quadraticAtt = 1.8f;   }
                    ImGui::SameLine();
                    if (ImGui::Button("20 метров")) { p.constantAtt = 1.0f; p.linearAtt = 0.22f;  p.quadraticAtt = 0.20f;  }
                    ImGui::SameLine();
                    if (ImGui::Button("50 метров")) { p.constantAtt = 1.0f; p.linearAtt = 0.09f;  p.quadraticAtt = 0.032f; }

                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        // --- 3. Spotlights (SSBO) ---
        if (ImGui::CollapsingHeader("3. Прожекторные источники (SSBO)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Активно прожекторов: %zu / %u", spotLights.size(), kMaxLights);
            if (spotLights.size() < kMaxLights) {
                if (ImGui::Button("+ Добавить прожектор")) {
                    graphics::SpotLight s;
                    s.name = "Прожектор #" + std::to_string(spotLights.size() + 1);
                    s.position = glm::vec3(2.0f, 3.5f, 2.0f);
                    s.direction = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f) - s.position);
                    s.color = glm::vec3(0.2f, 1.0f, 0.4f);
                    s.intensity = 3.0f;
                    s.innerAngleDegrees = 15.0f;
                    s.outerAngleDegrees = 25.0f;
                    spotLights.push_back(s);
                }
            }

            for (size_t i = 0; i < spotLights.size(); ++i) {
                auto& s = spotLights[i];
                ImGui::PushID(static_cast<int>(1000 + i));
                if (ImGui::TreeNode(s.name.c_str())) {
                    ImGui::Checkbox("Активен", &s.enabled);
                    ImGui::SameLine();
                    if (ImGui::Button("Удалить прожектор")) {
                        spotLights.erase(spotLights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }

                    ImGui::DragFloat3("Позиция", glm::value_ptr(s.position), 0.05f);
                    ImGui::DragFloat3("Направление луча", glm::value_ptr(s.direction), 0.02f, -1.0f, 1.0f);
                    if (ImGui::Button("Навести на тор")) {
                        s.direction = glm::normalize(torusObject.position - s.position);
                    }

                    ImGui::ColorEdit3("Цвет", glm::value_ptr(s.color));
                    ImGui::SliderFloat("Интенсивность", &s.intensity, 0.0f, 15.0f);

                    ImGui::SliderFloat("Внутренний угол", &s.innerAngleDegrees, 1.0f, 60.0f, "%.1f°");
                    ImGui::SliderFloat("Внешний угол (мягкий край)", &s.outerAngleDegrees, s.innerAngleDegrees, 80.0f, "%.1f°");

                    ImGui::TextDisabled("Затухание луча:");
                    ImGui::DragFloat("Линейный (kl)", &s.linearAtt, 0.005f, 0.0f, 1.0f, "%.4f");
                    ImGui::DragFloat("Квадратичный (kq)", &s.quadraticAtt, 0.001f, 0.0f, 0.5f, "%.5f");

                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
    }
    ImGui::End();

    // Window 3: Materials & Model Properties
    ImGui::SetNextWindowPos(ImVec2(850, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Материалы и Объекты")) {
        if (ImGui::CollapsingHeader("Тор (Вариант 7)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Геометрия тора:");
            if (ImGui::SliderFloat("Большой радиус (R)", &torusMajorRadius, 0.5f, 3.0f)) torusMeshDirty = true;
            if (ImGui::SliderFloat("Малый радиус (r)", &torusMinorRadius, 0.1f, 1.5f)) torusMeshDirty = true;
            if (ImGui::SliderInt("Сегментов по окружности", &torusRadialSegments, 8, 96)) torusMeshDirty = true;
            if (ImGui::SliderInt("Сегментов трубы", &torusTubularSegments, 6, 64)) torusMeshDirty = true;

            const char* colorModes[] = { "Локальные координаты (XYZ)", "Радужный UV", "Белый монохром" };
            if (ImGui::Combo("Режим процедурного цвета", &torusColorMode, colorModes, 3)) torusMeshDirty = true;

            ImGui::Separator();
            ImGui::Text("Материал Блинн-Фонга:");
            ImGui::ColorEdit4("Альбедо (Базовый цвет)", glm::value_ptr(torusObject.albedo));
            ImGui::ColorEdit3("Цвет блика (Specular)", glm::value_ptr(torusObject.specularColor));
            ImGui::SliderFloat("Коэффициент блика (Shininess)", &torusObject.shininess, 1.0f, 256.0f, "%.0f");
            ImGui::Checkbox("Умножать на процедурный цвет вершин", &torusObject.useVertexColor);

            ImGui::Separator();
            ImGui::Text("Трансформация тора:");
            ImGui::DragFloat3("Позиция##Torus", glm::value_ptr(torusObject.position), 0.05f);
            ImGui::DragFloat3("Поворот##Torus", glm::value_ptr(torusObject.rotation), 1.0f);
            ImGui::DragFloat3("Масштаб##Torus", glm::value_ptr(torusObject.scale), 0.02f, 0.1f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Пол (Сетка)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Отображать пол", &floorObject.visible);
            ImGui::ColorEdit4("Альбедо пола", glm::value_ptr(floorObject.albedo));
            ImGui::ColorEdit3("Цвет блика пола", glm::value_ptr(floorObject.specularColor));
            ImGui::SliderFloat("Блик пола (Shininess)", &floorObject.shininess, 1.0f, 128.0f);
        }
    }
    ImGui::End();

    // ------------------------------------------------------------------------
    // Update GPU Buffers
    // ------------------------------------------------------------------------
    auto& ctx = graphics::internal::context;
    float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                   static_cast<float>(ctx.swapchain_extent.height);

    // 1. Update Scene UBO
    SceneUBO scene{};
    scene.view = camera.getViewMatrix();
    scene.proj = camera.getProjectionMatrix(aspect);
    scene.cameraPos = glm::vec4(camera.position, 1.0f);
    scene.ambientColor = glm::vec4(ambientLightColor * ambientIntensity, 1.0f);

    if (dirLightEnabled) {
        scene.dirLightDir = glm::vec4(glm::normalize(dirLightDirection), 0.0f);
        scene.dirLightColor = glm::vec4(dirLightColor, dirLightIntensity);
    } else {
        scene.dirLightColor = glm::vec4(0.0f);
    }

    scene.numPointLights = static_cast<int32_t>(std::min(pointLights.size(), static_cast<size_t>(kMaxLights)));
    scene.numSpotLights = static_cast<int32_t>(std::min(spotLights.size(), static_cast<size_t>(kMaxLights)));

    std::memcpy(sceneUboBuffer.mapped, &scene, sizeof(scene));

    // 2. Update Point Lights SSBO
    if (pointLightSSBO.mapped) {
        std::vector<graphics::PointLightGPU> gpuPoints;
        gpuPoints.reserve(pointLights.size());
        for (const auto& pl : pointLights) {
            gpuPoints.push_back(pl.toGPU(animateLights ? currentTime : 0.0f));
        }
        if (!gpuPoints.empty()) {
            std::memcpy(pointLightSSBO.mapped, gpuPoints.data(),
                        sizeof(graphics::PointLightGPU) * gpuPoints.size());
        }
    }

    // 3. Update Spotlights SSBO
    if (spotLightSSBO.mapped) {
        std::vector<graphics::SpotLightGPU> gpuSpots;
        gpuSpots.reserve(spotLights.size());
        for (const auto& sl : spotLights) {
            gpuSpots.push_back(sl.toGPU(animateLights ? currentTime : 0.0f));
        }
        if (!gpuSpots.empty()) {
            std::memcpy(spotLightSSBO.mapped, gpuSpots.data(),
                        sizeof(graphics::SpotLightGPU) * gpuSpots.size());
        }
    }

    // 4. Update Model UBOs
    torusObject.updateUBO();
    floorObject.updateUBO();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    vkResetCommandBuffer(fd.command_buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.08f, 0.09f, 0.12f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBegin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = ctx.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = {
            .offset = {0, 0},
            .extent = ctx.swapchain_extent,
        },
        .clearValueCount = 2,
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(ctx.swapchain_extent.width),
        .height = static_cast<float>(ctx.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = ctx.swapchain_extent,
    };
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    // ------------------------------------------------------------------------
    // 1. Draw Main Scene Objects (Blinn-Phong)
    // ------------------------------------------------------------------------
    if (mainPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipeline);

        // Bind Scene Descriptor Set (Set 0) ONCE for the entire frame
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                mainPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);

        // Draw Models (Binding Set 1 per object)
        SceneObject* objects[] = { &torusObject, &floorObject };
        for (SceneObject* obj : objects) {
            if (!obj->visible || obj->indexCount == 0) continue;

            // Bind Model Descriptor Set (Set 1)
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    mainPipelineLayout, 1, 1, &obj->descriptorSet, 0, nullptr);

            VkBuffer vertexBuffers[] = { obj->vertexBuffer.buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(fd.command_buffer, obj->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(fd.command_buffer, obj->indexCount, 1, 0, 0, 0);
        }
    }

    // ------------------------------------------------------------------------
    // 2. Draw Light Visualizers (Task 4 for Grade 5)
    // ------------------------------------------------------------------------
    if (showLightVisualizers && unlitMarkerPipeline != VK_NULL_HANDLE && unlitPipelineLayout != VK_NULL_HANDLE) {
        // Bind Scene Set 0 to get View and Proj matrices
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                unlitPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);

        // --- Draw Point Light Markers (Bright Solid Cubes) ---
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitMarkerPipeline);
        VkBuffer vBuffers[] = { cubeVertexBuffer.buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vBuffers, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, cubeIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (const auto& pl : pointLights) {
            if (!pl.enabled) continue;
            glm::vec3 pos = pl.getActivePosition(animateLights ? currentTime : 0.0f);

            PushConstantBlock pc{};
            pc.model = glm::translate(glm::mat4(1.0f), pos);
            pc.color = glm::vec4(pl.color, 1.0f);

            vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndexed(fd.command_buffer, cubeIndexCount, 1, 0, 0, 0);
        }

        // --- Draw Spotlight Bases & Cone Lines ---
        for (const auto& sl : spotLights) {
            if (!sl.enabled) continue;
            glm::vec3 pos = sl.getActivePosition(animateLights ? currentTime : 0.0f);
            glm::vec3 dir = sl.getActiveDirection(animateLights ? currentTime : 0.0f);

            // Draw small cube at spotlight head
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitMarkerPipeline);
            PushConstantBlock pc{};
            pc.model = glm::translate(glm::mat4(1.0f), pos);
            pc.color = glm::vec4(sl.color, 1.0f);
            vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndexed(fd.command_buffer, cubeIndexCount, 1, 0, 0, 0);

            // Draw Wireframe Cone for Spotlight beam
            if (unlitConePipeline != VK_NULL_HANDLE && coneIndexCount > 0) {
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitConePipeline);
                VkBuffer cVBuffers[] = { coneVertexBuffer.buffer };
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, cVBuffers, offsets);
                vkCmdBindIndexBuffer(fd.command_buffer, coneIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

                // Compute rotation aligning default -Z axis with spotlight direction
                glm::vec3 defaultDir(0.0f, 0.0f, -1.0f);
                glm::mat4 coneModel(1.0f);
                coneModel = glm::translate(coneModel, pos);

                float d = glm::dot(defaultDir, dir);
                if (d < -0.9999f) {
                    coneModel = glm::rotate(coneModel, glm::radians(180.0f), glm::vec3(0, 1, 0));
                } else if (d < 0.9999f) {
                    glm::vec3 axis = glm::normalize(glm::cross(defaultDir, dir));
                    float angle = std::acos(glm::clamp(d, -1.0f, 1.0f));
                    coneModel = glm::rotate(coneModel, angle, axis);
                }

                // Scale cone by outer angle tan
                float angleScale = std::tan(glm::radians(sl.outerAngleDegrees)) / std::tan(glm::radians(25.0f));
                coneModel = glm::scale(coneModel, glm::vec3(angleScale, angleScale, 1.0f));

                PushConstantBlock conePc{};
                conePc.model = coneModel;
                conePc.color = glm::vec4(sl.color * 0.8f + glm::vec3(0.2f), 0.7f); // tinted white line

                vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(conePc), &conePc);
                vkCmdDrawIndexed(fd.command_buffer, coneIndexCount, 1, 0, 0, 0);
            }
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
