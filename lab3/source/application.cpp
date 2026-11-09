#include "application.hpp"
#include "camera.hpp"
#include "lights.hpp"
#include "geometry.hpp"
#include "texture.hpp"

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
    return "MAI Computer Graphics - Lab 3: Texture Mapping, Specular & Emissive";
}

namespace {

// ----------------------------------------------------------------------------
// GPU Structures (Matching Shaders)
// ----------------------------------------------------------------------------
struct alignas(16) SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 cameraPos;
    glm::vec4 ambientColor;
    glm::vec4 dirLightDir;
    glm::vec4 dirLightColor;
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
    glm::vec4 params;        // x = ambientMul, y = useVertexColor, zw = uvScale
};

struct alignas(16) PushConstants {
    int32_t nonTrivialSampling = 0; // 0 = standard, 1 = waves & multi-tap, 2 = dual-flow plasma
    float time = 0.0f;
    float effectSpeed = 1.0f;
    float effectScale = 1.0f;
    glm::vec4 colorModulation{1.0f, 1.0f, 1.0f, 1.0f};
    float emissiveMultiplier = 1.5f;
};

struct PushConstantBlockUnlit {
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

// ----------------------------------------------------------------------------
// GPU Commands Helper for VRAM Device-Local Staging Copy
// ----------------------------------------------------------------------------
VkCommandPool s_appTransferPool = VK_NULL_HANDLE;

VkCommandPool getAppTransferPool() {
    auto& ctx = graphics::internal::context;
    if (s_appTransferPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = ctx.graphics_queue_index,
        };
        vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &s_appTransferPool);
    }
    return s_appTransferPool;
}

VkCommandBuffer beginSingleTimeCommands() {
    auto& ctx = graphics::internal::context;
    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = getAppTransferPool(),
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void endSingleTimeCommands(VkCommandBuffer cmd) {
    auto& ctx = graphics::internal::context;
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(ctx.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphics_queue);

    vkFreeCommandBuffers(ctx.device, getAppTransferPool(), 1, &cmd);
}

// Upload buffer directly into fast GPU VRAM using Staging Buffer copy
bool createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data, BufferResource& outBuffer) {
    auto& ctx = graphics::internal::context;

    // 1. Host Staging Buffer
    BufferResource staging;
    if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, staging)) {
        return false;
    }
    std::memcpy(staging.mapped, data, size);

    // 2. Pure Device Local VRAM Buffer
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo allocInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo,
                        &outBuffer.buffer, &outBuffer.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to allocate VRAM device local buffer\n";
        destroyBuffer(staging);
        return false;
    }
    outBuffer.size = size;
    outBuffer.mapped = nullptr;

    // 3. GPU Copy Command: Staging -> VRAM
    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion = { .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer(cmd, staging.buffer, outBuffer.buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);

    destroyBuffer(staging);
    return true;
}

bool uploadMeshToVRAM(const graphics::Mesh& mesh, BufferResource& vb, BufferResource& ib, uint32_t& outIndexCount) {
    destroyBuffer(vb);
    destroyBuffer(ib);

    VkDeviceSize vSize = sizeof(graphics::Vertex) * mesh.vertices.size();
    VkDeviceSize iSize = sizeof(uint32_t) * mesh.indices.size();

    if (!createDeviceLocalBuffer(vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices.data(), vb)) {
        return false;
    }
    if (!createDeviceLocalBuffer(iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices.data(), ib)) {
        return false;
    }

    outIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return true;
}

std::vector<char> readFile(const std::string& filename) {
    const std::vector<std::string> candidates = {
        filename,
        "lab3/" + filename,
        "../" + filename,
        "../lab3/" + filename,
        "../../" + filename,
        "../../lab3/" + filename,
        "shaders/" + filename,
        "lab3/shaders/" + filename,
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
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

// ----------------------------------------------------------------------------
// Material Definition (Set 1 with Textures & Shared Dynamic UBO)
// ----------------------------------------------------------------------------
struct Material {
    std::string name;
    graphics::Texture albedoTex;
    graphics::Texture specularTex;
    graphics::Texture emissiveTex;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

// ----------------------------------------------------------------------------
// Scene Object Definition
// ----------------------------------------------------------------------------
struct SceneObject {
    std::string name;
    bool visible = true;
    uint32_t modelIndex = 0; // Index in the single Dynamic Uniform Buffer

    // GPU buffers in VRAM
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    uint32_t indexCount = 0;

    // Active material index
    int selectedMaterial = 0;

    // Transform
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 rotation{0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    // Material parameters
    glm::vec4 albedo{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec3 specularColor{1.0f, 1.0f, 1.0f};
    float shininess = 64.0f;
    float ambientMultiplier = 1.0f;
    bool useVertexColor = false;
    glm::vec2 uvScale{1.0f, 1.0f};

    // Non-trivial sampling settings (Task 2 for Grade 4)
    PushConstants pushConstants;

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
};

// ----------------------------------------------------------------------------
// Global Application State
// ----------------------------------------------------------------------------
graphics::Camera camera;

// Lighting State
glm::vec3 ambientLightColor{0.15f, 0.15f, 0.18f};
float ambientIntensity = 1.0f;

bool dirLightEnabled = true;
glm::vec3 dirLightDirection{0.5f, 1.0f, 0.3f};
glm::vec3 dirLightColor{1.0f, 0.95f, 0.85f};
float dirLightIntensity = 1.2f;

std::vector<graphics::PointLight> pointLights;
std::vector<graphics::SpotLight> spotLights;

constexpr uint32_t kMaxLights = 64;
BufferResource pointLightSSBO;
BufferResource spotLightSSBO;

// Scene UBO (Set 0)
BufferResource sceneUboBuffer;

// Dynamic Uniform Buffer for All Models (Core requirement of Lab 3)
BufferResource dynamicModelBuffer;
VkDeviceSize alignedModelUboSize = 0;
constexpr uint32_t kMaxModels = 16;

// Materials list (Task 1 & 3)
std::vector<Material> materials;

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

// Vulkan Descriptors & Pipelines
VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelMaterialSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;

VkPipelineLayout mainPipelineLayout = VK_NULL_HANDLE;
VkPipeline mainPipeline = VK_NULL_HANDLE;

// Visualizer pipelines
VkPipelineLayout unlitPipelineLayout = VK_NULL_HANDLE;
VkPipeline unlitMarkerPipeline = VK_NULL_HANDLE;
VkPipeline unlitConePipeline = VK_NULL_HANDLE;
BufferResource cubeVertexBuffer, cubeIndexBuffer;
uint32_t cubeIndexCount = 0;
BufferResource coneVertexBuffer, coneIndexBuffer;
uint32_t coneIndexCount = 0;

bool showLightVisualizers = true;
bool animateLights = true;
float currentTime = 0.0f;

// ----------------------------------------------------------------------------
// Initialization Helpers
// ----------------------------------------------------------------------------
void rebuildTorusMesh() {
    auto mesh = graphics::createTorus(torusMajorRadius, torusMinorRadius,
                                      torusRadialSegments, torusTubularSegments,
                                      torusColorMode);
    uploadMeshToVRAM(mesh, torusObject.vertexBuffer, torusObject.indexBuffer, torusObject.indexCount);
    torusMeshDirty = false;
}

bool initDescriptorsAndDynamicUBO() {
    auto& ctx = graphics::internal::context;

    // Calculate Dynamic Uniform Buffer Alignment
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
    VkDeviceSize minAlign = props.limits.minUniformBufferOffsetAlignment;
    alignedModelUboSize = (sizeof(ModelUBO) + minAlign - 1) & ~(minAlign - 1);

    // 1. Allocate Single Dynamic Model Uniform Buffer
    VkDeviceSize dynamicBufferSize = alignedModelUboSize * kMaxModels;
    if (!createBuffer(dynamicBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      dynamicModelBuffer)) {
        return false;
    }

    // 2. Set 0 (Scene Layout)
    VkDescriptorSetLayoutBinding sceneBindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
    };
    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = sceneBindings,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr, &sceneSetLayout) != VK_SUCCESS) {
        return false;
    }

    // 3. Set 1 (Model Material Layout)
    // Binding 0: DYNAMIC UNIFORM BUFFER
    // Binding 1: Albedo sampler
    // Binding 2: Specular sampler
    // Binding 3: Emissive sampler
    VkDescriptorSetLayoutBinding modelBindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
    };
    VkDescriptorSetLayoutCreateInfo modelLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = modelBindings,
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &modelLayoutInfo, nullptr, &modelMaterialSetLayout) != VK_SUCCESS) {
        return false;
    }

    // 4. Descriptor Pool
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32 },
    };
    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 24,
        .poolSizeCount = 4,
        .pPoolSizes = poolSizes,
    };
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return false;
    }

    // 5. Allocate & Write Scene Descriptor Set
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &sceneSetLayout,
    };
    vkAllocateDescriptorSets(ctx.device, &allocInfo, &sceneDescriptorSet);

    VkDescriptorBufferInfo sceneUboInfo = { sceneUboBuffer.buffer, 0, sizeof(SceneUBO) };
    VkDescriptorBufferInfo pointSsboInfo = { pointLightSSBO.buffer, 0, pointLightSSBO.size };
    VkDescriptorBufferInfo spotSsboInfo = { spotLightSSBO.buffer, 0, spotLightSSBO.size };

    VkWriteDescriptorSet sceneWrites[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneUboInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pointSsboInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spotSsboInfo, nullptr },
    };
    vkUpdateDescriptorSets(ctx.device, 3, sceneWrites, 0, nullptr);

    return true;
}

bool createMaterialDescriptorSet(Material& mat) {
    auto& ctx = graphics::internal::context;

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &modelMaterialSetLayout,
    };
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &mat.descriptorSet) != VK_SUCCESS) {
        return false;
    }

    // Dynamic buffer info: range is single aligned structure
    VkDescriptorBufferInfo modelInfo = {
        .buffer = dynamicModelBuffer.buffer,
        .offset = 0,
        .range = sizeof(ModelUBO),
    };

    VkDescriptorImageInfo albedoInfo = {
        .sampler = mat.albedoTex.sampler,
        .imageView = mat.albedoTex.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo specInfo = {
        .sampler = mat.specularTex.sampler,
        .imageView = mat.specularTex.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo emissiveInfo = {
        .sampler = mat.emissiveTex.sampler,
        .imageView = mat.emissiveTex.imageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &modelInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albedoInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &specInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, mat.descriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &emissiveInfo, nullptr, nullptr },
    };
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);
    return true;
}

bool initPipelines() {
    auto& ctx = graphics::internal::context;

    // 1. Main Pipeline Layout with Push Constants
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    VkDescriptorSetLayout setLayouts[] = { sceneSetLayout, modelMaterialSetLayout };
    VkPipelineLayoutCreateInfo mainLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = setLayouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    if (vkCreatePipelineLayout(ctx.device, &mainLayoutInfo, nullptr, &mainPipelineLayout) != VK_SUCCESS) {
        return false;
    }

    auto vertCode = readFile("shaders/shader.vert.spv");
    auto fragCode = readFile("shaders/shader.frag.spv");
    if (vertCode.empty() || fragCode.empty()) return false;

    VkShaderModule vertModule = createShaderModule(ctx.device, vertCode);
    VkShaderModule fragModule = createShaderModule(ctx.device, fragCode);

    VkPipelineShaderStageCreateInfo stages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr },
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
        .cullMode = VK_CULL_MODE_NONE,
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
        .colorWriteMask = 0xF,
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
        .pStages = stages,
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
    vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mainPipeline);

    vkDestroyShaderModule(ctx.device, vertModule, nullptr);
    vkDestroyShaderModule(ctx.device, fragModule, nullptr);

    // 2. Unlit Visualizer Pipeline
    VkPushConstantRange unlitPcRange = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushConstantBlockUnlit),
    };
    VkPipelineLayoutCreateInfo unlitLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &sceneSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &unlitPcRange,
    };
    vkCreatePipelineLayout(ctx.device, &unlitLayoutInfo, nullptr, &unlitPipelineLayout);

    auto uVertCode = readFile("shaders/unlit.vert.spv");
    auto uFragCode = readFile("shaders/unlit.frag.spv");
    if (!uVertCode.empty() && !uFragCode.empty()) {
        VkShaderModule uVMod = createShaderModule(ctx.device, uVertCode);
        VkShaderModule uFMod = createShaderModule(ctx.device, uFragCode);

        VkPipelineShaderStageCreateInfo uStages[] = {
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, uVMod, "main", nullptr },
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, uFMod, "main", nullptr },
        };
        pipelineInfo.pStages = uStages;
        pipelineInfo.layout = unlitPipelineLayout;
        depthStencil.depthWriteEnable = VK_TRUE;
        pipelineInfo.pDepthStencilState = &depthStencil;
        vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &unlitMarkerPipeline);

        VkPipelineInputAssemblyStateCreateInfo lineAssembly = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        };
        pipelineInfo.pInputAssemblyState = &lineAssembly;
        depthStencil.depthWriteEnable = VK_FALSE;
        pipelineInfo.pDepthStencilState = &depthStencil;
        vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &unlitConePipeline);

        vkDestroyShaderModule(ctx.device, uVMod, nullptr);
        vkDestroyShaderModule(ctx.device, uFMod, nullptr);
    }
    return true;
}

} // namespace

// ----------------------------------------------------------------------------
// Public Application Interface
// ----------------------------------------------------------------------------
bool initialize() {
    // 1. Allocate Scene Uniform Buffer & SSBOs
    if (!createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      sceneUboBuffer)) {
        return false;
    }
    if (!createBuffer(sizeof(graphics::PointLightGPU) * kMaxLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, pointLightSSBO)) {
        return false;
    }
    if (!createBuffer(sizeof(graphics::SpotLightGPU) * kMaxLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, spotLightSSBO)) {
        return false;
    }

    // 2. Initialize Texture Manager
    if (!graphics::TextureManager::initialize()) {
        std::cerr << "Failed to initialize Texture Manager\n";
        return false;
    }

    // 3. Setup Descriptors, Dynamic UBO & Pipelines
    if (!initDescriptorsAndDynamicUBO()) return false;
    if (!initPipelines()) return false;

    // 4. Load Textures & Build Material Catalog (Task 1 & Task 3 for Grade 5)
    graphics::Texture texTechAlb, texTechSpec, texNeonEmi, texLavaEmi;
    graphics::Texture texTilesAlb, texTilesSpec;
    graphics::Texture texChecker;

    graphics::TextureManager::loadFromFile("textures/albedo_tech.png", "Tech Albedo", texTechAlb);
    graphics::TextureManager::loadFromFile("textures/specular_tech.png", "Tech Specular", texTechSpec);
    graphics::TextureManager::loadFromFile("textures/emissive_neon.png", "Neon Emissive", texNeonEmi);
    graphics::TextureManager::loadFromFile("textures/emissive_lava.png", "Lava Emissive", texLavaEmi);

    graphics::TextureManager::loadFromFile("textures/albedo_tiles.png", "Tiles Albedo", texTilesAlb);
    graphics::TextureManager::loadFromFile("textures/specular_tiles.png", "Tiles Specular", texTilesSpec);

    graphics::TextureManager::loadFromFile("textures/albedo_checker.png", "Checker Albedo", texChecker);

    // Material 0: Cyber Tech (Albedo + Specular + Neon Emissive)
    {
        Material m;
        m.name = "1. Cyber Tech (Неон)";
        m.albedoTex = texTechAlb.isValid() ? texTechAlb : graphics::TextureManager::whiteFallback;
        m.specularTex = texTechSpec.isValid() ? texTechSpec : graphics::TextureManager::whiteFallback;
        m.emissiveTex = texNeonEmi.isValid() ? texNeonEmi : graphics::TextureManager::blackFallback;
        createMaterialDescriptorSet(m);
        materials.push_back(m);
    }
    // Material 1: Magma Core (Albedo + Specular + Lava Emissive)
    {
        Material m;
        m.name = "2. Magma Core (Лава)";
        m.albedoTex = texTechAlb.isValid() ? texTechAlb : graphics::TextureManager::whiteFallback;
        m.specularTex = texTechSpec.isValid() ? texTechSpec : graphics::TextureManager::whiteFallback;
        m.emissiveTex = texLavaEmi.isValid() ? texLavaEmi : graphics::TextureManager::blackFallback;
        createMaterialDescriptorSet(m);
        materials.push_back(m);
    }
    // Material 2: Tiles & Marble (Albedo + Specular + Black Fallback for Emissive)
    {
        Material m;
        m.name = "3. Плитка (Без свечения)";
        m.albedoTex = texTilesAlb.isValid() ? texTilesAlb : graphics::TextureManager::whiteFallback;
        m.specularTex = texTilesSpec.isValid() ? texTilesSpec : graphics::TextureManager::whiteFallback;
        m.emissiveTex = graphics::TextureManager::blackFallback; // Uses 1x1 black fallback!
        createMaterialDescriptorSet(m);
        materials.push_back(m);
    }
    // Material 3: Checkerboard (Albedo + White Fallback + Black Fallback)
    {
        Material m;
        m.name = "4. Шахматка (Checkerboard)";
        m.albedoTex = texChecker.isValid() ? texChecker : graphics::TextureManager::whiteFallback;
        m.specularTex = graphics::TextureManager::whiteFallback;
        m.emissiveTex = graphics::TextureManager::blackFallback;
        createMaterialDescriptorSet(m);
        materials.push_back(m);
    }

    // 5. Create Scene Objects (Toroid & Floor Plane) in VRAM
    torusObject.name = "Тор (Вариант 7)";
    torusObject.modelIndex = 0;
    torusObject.position = glm::vec3(0.0f, 1.0f, 0.0f);
    torusObject.selectedMaterial = 0; // Default Cyber Tech
    torusObject.pushConstants.emissiveMultiplier = 2.0f;
    rebuildTorusMesh();

    floorObject.name = "Пол (Сетка)";
    floorObject.modelIndex = 1;
    floorObject.position = glm::vec3(0.0f, 0.0f, 0.0f);
    floorObject.selectedMaterial = 2; // Default Tiles
    floorObject.uvScale = glm::vec2(4.0f, 4.0f);
    auto floorMesh = graphics::createPlane(24.0f, 24.0f, 40, 40, 6.0f);
    uploadMeshToVRAM(floorMesh, floorObject.vertexBuffer, floorObject.indexBuffer, floorObject.indexCount);

    // 6. Visualizer Meshes
    auto cubeMesh = graphics::createCube(0.2f);
    uploadMeshToVRAM(cubeMesh, cubeVertexBuffer, cubeIndexBuffer, cubeIndexCount);
    auto coneMesh = graphics::createConeLines(2.5f, 25.0f, 32);
    uploadMeshToVRAM(coneMesh, coneVertexBuffer, coneIndexBuffer, coneIndexCount);

    // 7. Initial Lights
    {
        graphics::PointLight p1;
        p1.name = "Точечный #1 (Оранжевый)";
        p1.position = glm::vec3(2.5f, 1.8f, 0.0f);
        p1.color = glm::vec3(1.0f, 0.5f, 0.1f);
        p1.intensity = 2.0f;
        p1.animate = true;
        p1.animRadius = 3.0f;
        p1.animSpeed = 1.0f;
        pointLights.push_back(p1);
    }
    {
        graphics::PointLight p2;
        p2.name = "Точечный #2 (Бирюзовый)";
        p2.position = glm::vec3(-2.5f, 1.8f, 0.0f);
        p2.color = glm::vec3(0.1f, 0.8f, 1.0f);
        p2.intensity = 2.0f;
        p2.animate = true;
        p2.animRadius = 3.0f;
        p2.animSpeed = 1.0f;
        p2.animPhase = 3.14159f;
        pointLights.push_back(p2);
    }
    {
        graphics::SpotLight s1;
        s1.name = "Прожектор #1 (Синий)";
        s1.position = glm::vec3(0.0f, 4.0f, 2.5f);
        s1.direction = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f) - s1.position);
        s1.color = glm::vec3(0.4f, 0.7f, 1.0f);
        s1.intensity = 3.5f;
        s1.innerAngleDegrees = 15.0f;
        s1.outerAngleDegrees = 25.0f;
        spotLights.push_back(s1);
    }

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);

    for (auto& m : materials) {
        graphics::TextureManager::destroyTexture(m.albedoTex);
        graphics::TextureManager::destroyTexture(m.specularTex);
        graphics::TextureManager::destroyTexture(m.emissiveTex);
    }
    materials.clear();
    graphics::TextureManager::shutdown();

    destroyBuffer(cubeVertexBuffer);
    destroyBuffer(cubeIndexBuffer);
    destroyBuffer(coneVertexBuffer);
    destroyBuffer(coneIndexBuffer);

    destroyBuffer(torusObject.vertexBuffer);
    destroyBuffer(torusObject.indexBuffer);
    destroyBuffer(floorObject.vertexBuffer);
    destroyBuffer(floorObject.indexBuffer);

    destroyBuffer(dynamicModelBuffer);
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
    if (modelMaterialSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, modelMaterialSetLayout, nullptr);

    if (s_appTransferPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx.device, s_appTransferPool, nullptr);
        s_appTransferPool = VK_NULL_HANDLE;
    }
}

void update(double time) {
    currentTime = static_cast<float>(time);
    torusObject.pushConstants.time = currentTime;
    floorObject.pushConstants.time = currentTime;

    if (torusMeshDirty) {
        rebuildTorusMesh();
    }

    // Camera Navigation (Hold Right Mouse Button)
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
    // ImGUI Controls
    // ------------------------------------------------------------------------
    // Window 1: Textures & Materials (Task 1, 2, 3)
    ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(440, 680), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Текстурирование (Лаб 3)")) {
        // Prepare list of material names
        std::vector<const char*> matNames;
        for (const auto& m : materials) matNames.push_back(m.name.c_str());

        // --- Torus Material & Sampling ---
        if (ImGui::CollapsingHeader("1. Тор (Вариант 7) - Текстура и Эффекты", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Combo("Выбор материала / текстур", &torusObject.selectedMaterial, matNames.data(), static_cast<int>(matNames.size()));

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.3f, 1.0f), "Нетривиальное сэмплирование (Push Constants):");
            const char* samplingModes[] = {
                "0: Стандартное наложение",
                "1: Волновые искажения + мультисэмплирование",
                "2: Двухслойная динамическая плазма"
            };
            ImGui::Combo("Режим сэмплирования", &torusObject.pushConstants.nonTrivialSampling, samplingModes, 3);
            if (torusObject.pushConstants.nonTrivialSampling > 0) {
                ImGui::SliderFloat("Скорость анимации UV", &torusObject.pushConstants.effectSpeed, 0.1f, 5.0f);
                ImGui::SliderFloat("Сила искажения", &torusObject.pushConstants.effectScale, 0.1f, 3.0f);
                ImGui::ColorEdit3("Модуляция цвета", glm::value_ptr(torusObject.pushConstants.colorModulation));
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "Specular & Emissive (Оценка 5):");
            ImGui::SliderFloat("Яркость свечения (Emissive)", &torusObject.pushConstants.emissiveMultiplier, 0.0f, 5.0f, "%.2f");
            ImGui::ColorEdit3("Цвет блика (Specular)", glm::value_ptr(torusObject.specularColor));
            ImGui::SliderFloat("Коэффициент блика (Shininess)", &torusObject.shininess, 1.0f, 256.0f);

            ImGui::Separator();
            ImGui::Text("Геометрия тора:");
            if (ImGui::SliderFloat("Большой радиус (R)", &torusMajorRadius, 0.5f, 3.0f)) torusMeshDirty = true;
            if (ImGui::SliderFloat("Малый радиус (r)", &torusMinorRadius, 0.1f, 1.5f)) torusMeshDirty = true;
            if (ImGui::SliderInt("Сегментов R", &torusRadialSegments, 8, 96)) torusMeshDirty = true;
            if (ImGui::SliderInt("Сегментов r", &torusTubularSegments, 6, 64)) torusMeshDirty = true;
        }

        // --- Floor Material ---
        if (ImGui::CollapsingHeader("2. Пол (Сетка)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Отображать пол", &floorObject.visible);
            ImGui::Combo("Материал пола", &floorObject.selectedMaterial, matNames.data(), static_cast<int>(matNames.size()));
            ImGui::SliderFloat2("Тайлинг текстуры пола", glm::value_ptr(floorObject.uvScale), 1.0f, 10.0f);
        }

        // --- Info on Memory Architecture ---
        if (ImGui::CollapsingHeader("Архитектура памяти (VRAM & Dynamic UBO)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BulletText("Вершины и индексы: в VRAM (Device Local)");
            ImGui::BulletText("Копирование геометрии: Staging Buffer + vkCmdCopyBuffer");
            ImGui::BulletText("Свойства моделей: единый Dynamic Uniform Buffer");
            ImGui::BulletText("Выравнивание динамического UBO: %llu байт", static_cast<unsigned long long>(alignedModelUboSize));
            ImGui::BulletText("Управление сэмплированием: через Push Constants");
        }
    }
    ImGui::End();

    // Window 2: Camera & Lights
    ImGui::SetNextWindowPos(ImVec2(470, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 680), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Камера и Освещение")) {
        if (ImGui::CollapsingHeader("Камера", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Позиция", glm::value_ptr(camera.position), 0.05f);
            ImGui::SliderFloat("Поворот (Yaw)", &camera.yaw, -180.0f, 180.0f, "%.1f°");
            ImGui::SliderFloat("Наклон (Pitch)", &camera.pitch, -89.0f, 89.0f, "%.1f°");
            ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 110.0f, "%.1f°");
            if (ImGui::Button("Сброс камеры")) camera.resetToDefault();
            ImGui::SameLine();
            ImGui::Checkbox("Визуализаторы света", &showLightVisualizers);
            ImGui::SameLine();
            ImGui::Checkbox("Анимация ламп", &animateLights);
        }

        if (ImGui::CollapsingHeader("Рассеянный свет (Ambient)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Цвет рассеяния", glm::value_ptr(ambientLightColor));
            ImGui::SliderFloat("Интенсивность", &ambientIntensity, 0.0f, 2.0f);
        }

        if (ImGui::CollapsingHeader("Направленный свет", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Включен##Dir", &dirLightEnabled);
            ImGui::SliderFloat3("Направление##Dir", glm::value_ptr(dirLightDirection), -1.0f, 1.0f);
            ImGui::ColorEdit3("Цвет##Dir", glm::value_ptr(dirLightColor));
            ImGui::SliderFloat("Интенсивность##Dir", &dirLightIntensity, 0.0f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Точечные источники (SSBO)")) {
            if (ImGui::Button("+ Добавить точечный")) {
                graphics::PointLight p;
                p.name = "Точечный #" + std::to_string(pointLights.size() + 1);
                p.position = glm::vec3(0.0f, 2.0f, 1.5f);
                p.color = glm::vec3(1.0f, 0.8f, 0.3f);
                p.intensity = 2.0f;
                pointLights.push_back(p);
            }
            for (size_t i = 0; i < pointLights.size(); ++i) {
                auto& p = pointLights[i];
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode(p.name.c_str())) {
                    ImGui::Checkbox("Активен", &p.enabled);
                    ImGui::SameLine();
                    if (ImGui::Button("Удалить")) {
                        pointLights.erase(pointLights.begin() + i);
                        ImGui::TreePop(); ImGui::PopID(); break;
                    }
                    ImGui::DragFloat3("Позиция", glm::value_ptr(p.position), 0.05f);
                    ImGui::ColorEdit3("Цвет", glm::value_ptr(p.color));
                    ImGui::SliderFloat("Интенсивность", &p.intensity, 0.0f, 10.0f);
                    ImGui::Checkbox("Орбита", &p.animate);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Прожекторы (SSBO)")) {
            if (ImGui::Button("+ Добавить прожектор")) {
                graphics::SpotLight s;
                s.name = "Прожектор #" + std::to_string(spotLights.size() + 1);
                s.position = glm::vec3(2.0f, 3.5f, 2.0f);
                s.direction = glm::normalize(torusObject.position - s.position);
                s.color = glm::vec3(0.2f, 1.0f, 0.5f);
                s.intensity = 3.0f;
                spotLights.push_back(s);
            }
            for (size_t i = 0; i < spotLights.size(); ++i) {
                auto& s = spotLights[i];
                ImGui::PushID(static_cast<int>(500 + i));
                if (ImGui::TreeNode(s.name.c_str())) {
                    ImGui::Checkbox("Активен", &s.enabled);
                    ImGui::SameLine();
                    if (ImGui::Button("Удалить")) {
                        spotLights.erase(spotLights.begin() + i);
                        ImGui::TreePop(); ImGui::PopID(); break;
                    }
                    ImGui::DragFloat3("Позиция", glm::value_ptr(s.position), 0.05f);
                    ImGui::DragFloat3("Направление", glm::value_ptr(s.direction), 0.02f);
                    ImGui::ColorEdit3("Цвет", glm::value_ptr(s.color));
                    ImGui::SliderFloat("Интенсивность", &s.intensity, 0.0f, 15.0f);
                    ImGui::SliderFloat("Внутренний угол", &s.innerAngleDegrees, 1.0f, 60.0f, "%.1f°");
                    ImGui::SliderFloat("Внешний угол", &s.outerAngleDegrees, s.innerAngleDegrees, 80.0f, "%.1f°");
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
    }
    ImGui::End();

    // ------------------------------------------------------------------------
    // Update GPU Buffers
    // ------------------------------------------------------------------------
    auto& ctx = graphics::internal::context;
    float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                   static_cast<float>(ctx.swapchain_extent.height);

    // 1. Scene UBO
    SceneUBO scene{};
    scene.view = camera.getViewMatrix();
    scene.proj = camera.getProjectionMatrix(aspect);
    scene.cameraPos = glm::vec4(camera.position, 1.0f);
    scene.ambientColor = glm::vec4(ambientLightColor * ambientIntensity, 1.0f);
    if (dirLightEnabled) {
        scene.dirLightDir = glm::vec4(glm::normalize(dirLightDirection), 0.0f);
        scene.dirLightColor = glm::vec4(dirLightColor, dirLightIntensity);
    }
    scene.numPointLights = static_cast<int32_t>(std::min(pointLights.size(), static_cast<size_t>(kMaxLights)));
    scene.numSpotLights = static_cast<int32_t>(std::min(spotLights.size(), static_cast<size_t>(kMaxLights)));
    std::memcpy(sceneUboBuffer.mapped, &scene, sizeof(scene));

    // 2. Point Lights SSBO
    if (pointLightSSBO.mapped) {
        std::vector<graphics::PointLightGPU> gpuPoints;
        for (const auto& pl : pointLights) {
            gpuPoints.push_back(pl.toGPU(animateLights ? currentTime : 0.0f));
        }
        if (!gpuPoints.empty()) {
            std::memcpy(pointLightSSBO.mapped, gpuPoints.data(),
                        sizeof(graphics::PointLightGPU) * gpuPoints.size());
        }
    }

    // 3. Spotlights SSBO
    if (spotLightSSBO.mapped) {
        std::vector<graphics::SpotLightGPU> gpuSpots;
        for (const auto& sl : spotLights) {
            gpuSpots.push_back(sl.toGPU(animateLights ? currentTime : 0.0f));
        }
        if (!gpuSpots.empty()) {
            std::memcpy(spotLightSSBO.mapped, gpuSpots.data(),
                        sizeof(graphics::SpotLightGPU) * gpuSpots.size());
        }
    }

    // 4. Update Single Dynamic Model Uniform Buffer
    SceneObject* objects[] = { &torusObject, &floorObject };
    for (SceneObject* obj : objects) {
        ModelUBO ubo{};
        ubo.model = obj->computeModelMatrix();
        ubo.normalMatrix = obj->computeNormalMatrix();
        ubo.albedo = obj->albedo;
        ubo.specularColor = glm::vec4(obj->specularColor, obj->shininess);
        ubo.params = glm::vec4(obj->ambientMultiplier, obj->useVertexColor ? 1.0f : 0.0f,
                               obj->uvScale.x, obj->uvScale.y);

        char* dst = static_cast<char*>(dynamicModelBuffer.mapped) + (obj->modelIndex * alignedModelUboSize);
        std::memcpy(dst, &ubo, sizeof(ModelUBO));
    }
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
    clearValues[0].color = {{0.07f, 0.08f, 0.11f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBegin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = ctx.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .offset = {0, 0}, .extent = ctx.swapchain_extent },
        .clearValueCount = 2,
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {
        .x = 0.0f, .y = 0.0f,
        .width = static_cast<float>(ctx.swapchain_extent.width),
        .height = static_cast<float>(ctx.swapchain_extent.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    VkRect2D scissor = { .offset = {0, 0}, .extent = ctx.swapchain_extent };
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    // ------------------------------------------------------------------------
    // 1. Draw Main Textured Scene Objects
    // ------------------------------------------------------------------------
    if (mainPipeline != VK_NULL_HANDLE && !materials.empty()) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPipeline);

        // Bind Scene Descriptor Set (Set 0) ONCE for the whole frame
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                mainPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);

        SceneObject* objects[] = { &torusObject, &floorObject };
        for (SceneObject* obj : objects) {
            if (!obj->visible || obj->indexCount == 0) continue;

            // Push Constants for sampling effects & emissive multiplier
            vkCmdPushConstants(fd.command_buffer, mainPipelineLayout,
                               VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(PushConstants), &obj->pushConstants);

            // Bind Material Descriptor Set (Set 1) with DYNAMIC UNIFORM OFFSET
            uint32_t dynamicOffset = static_cast<uint32_t>(obj->modelIndex * alignedModelUboSize);
            size_t matIdx = static_cast<size_t>(std::clamp(obj->selectedMaterial, 0, static_cast<int>(materials.size() - 1)));

            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    mainPipelineLayout, 1, 1, &materials[matIdx].descriptorSet,
                                    1, &dynamicOffset);

            VkBuffer vBuffers[] = { obj->vertexBuffer.buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vBuffers, offsets);
            vkCmdBindIndexBuffer(fd.command_buffer, obj->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(fd.command_buffer, obj->indexCount, 1, 0, 0, 0);
        }
    }

    // ------------------------------------------------------------------------
    // 2. Draw Light Visualizers (Task 4 for Grade 5)
    // ------------------------------------------------------------------------
    if (showLightVisualizers && unlitMarkerPipeline != VK_NULL_HANDLE && unlitPipelineLayout != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                unlitPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);

        // Point Light Markers (Bright Solid Cubes)
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitMarkerPipeline);
        VkBuffer vBuffers[] = { cubeVertexBuffer.buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vBuffers, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, cubeIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (const auto& pl : pointLights) {
            if (!pl.enabled) continue;
            glm::vec3 pos = pl.getActivePosition(animateLights ? currentTime : 0.0f);

            PushConstantBlockUnlit pc{};
            pc.model = glm::translate(glm::mat4(1.0f), pos);
            pc.color = glm::vec4(pl.color, 1.0f);

            vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndexed(fd.command_buffer, cubeIndexCount, 1, 0, 0, 0);
        }

        // Spotlight Markers & Cone Lines
        for (const auto& sl : spotLights) {
            if (!sl.enabled) continue;
            glm::vec3 pos = sl.getActivePosition(animateLights ? currentTime : 0.0f);
            glm::vec3 dir = sl.getActiveDirection(animateLights ? currentTime : 0.0f);

            // Cube head
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitMarkerPipeline);
            PushConstantBlockUnlit pc{};
            pc.model = glm::translate(glm::mat4(1.0f), pos);
            pc.color = glm::vec4(sl.color, 1.0f);
            vkCmdPushConstants(fd.command_buffer, unlitPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndexed(fd.command_buffer, cubeIndexCount, 1, 0, 0, 0);

            // Cone lines
            if (unlitConePipeline != VK_NULL_HANDLE && coneIndexCount > 0) {
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, unlitConePipeline);
                VkBuffer cVBuffers[] = { coneVertexBuffer.buffer };
                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, cVBuffers, offsets);
                vkCmdBindIndexBuffer(fd.command_buffer, coneIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

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

                float angleScale = std::tan(glm::radians(sl.outerAngleDegrees)) / std::tan(glm::radians(25.0f));
                coneModel = glm::scale(coneModel, glm::vec3(angleScale, angleScale, 1.0f));

                PushConstantBlockUnlit conePc{};
                conePc.model = coneModel;
                conePc.color = glm::vec4(sl.color * 0.8f + glm::vec3(0.2f), 0.7f);

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
