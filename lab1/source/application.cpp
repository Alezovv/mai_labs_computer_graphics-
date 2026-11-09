#include "application.hpp"
#include "torus.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <imgui.h>

namespace application {

const char* getTitle() {
    return "MAI Computer Graphics - Lab 1: Torus";
}

namespace {

// Uniform Buffer Object layout matching shader
struct alignas(16) UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 objectColor;
    glm::vec4 lightDir;
    float ambientStrength;
    float specularStrength;
    float shininess;
    float pad;
};

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

    VmaAllocationInfo allocationInfo{};
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo,
                        &outBuffer.buffer, &outBuffer.allocation,
                        &allocationInfo) != VK_SUCCESS) {
        std::cerr << "Failed to allocate Vulkan buffer of size " << size << '\n';
        return false;
    }

    outBuffer.size = size;
    outBuffer.mapped = allocationInfo.pMappedData;
    return true;
}

void destroyBuffer(BufferResource& buf) {
    if (buf.buffer != VK_NULL_HANDLE) {
        auto& ctx = graphics::internal::context;
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
        "lab1/" + filename,
        "../" + filename,
        "../lab1/" + filename,
        "../../" + filename,
        "../../lab1/" + filename,
        "shaders/" + filename,
        "lab1/shaders/" + filename,
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
    std::cerr << "Could not open file: " << filename << '\n';
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

// Scene Object (represents a Torus with its own buffers, transform, animation and VkDescriptorSet)
struct TorusObject {
    std::string name;
    bool visible = true;

    // Torus geometry parameters
    float majorRadius = 1.5f;
    float minorRadius = 0.5f;
    int radialSegments = 48;
    int tubularSegments = 24;
    int colorMode = 0; // 0: Local XYZ->RGB, 1: Rainbow UV, 2: Normals, 3: Checkered
    bool meshDirty = false;

    // GPU mesh buffers
    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    uint32_t indexCount = 0;

    // Transform parameters (Task 2)
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 rotation{0.0f, 0.0f, 0.0f}; // Euler angles in degrees
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    bool uniformScale = true;

    // Material color (Task 4)
    glm::vec4 objectColor{1.0f, 0.85f, 0.35f, 1.0f};

    // Animation / Complex trajectory (Task 3)
    bool animate = false;
    int trajectoryType = 0; // 0: Lissajous 3D, 1: Trefoil Knot, 2: Spatial Spiral, 3: Lemniscate
    float trajectorySpeed = 1.0f;
    float trajectoryRadius = 3.0f;
    float trajectoryHeight = 1.5f;
    bool autoRotate = true;
    glm::vec3 rotationSpeed{30.0f, 45.0f, 20.0f}; // deg/sec
    float animTime = 0.0f;

    // Task 6: Individual Uniform Buffer and Descriptor Set per object
    BufferResource uniformBuffer;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    bool buildMesh() {
        graphics::TorusMesh mesh = graphics::TorusMesh::generate(
            majorRadius, minorRadius,
            static_cast<uint32_t>(radialSegments),
            static_cast<uint32_t>(tubularSegments),
            colorMode
        );

        destroyBuffer(vertexBuffer);
        destroyBuffer(indexBuffer);

        VkDeviceSize vSize = sizeof(graphics::Vertex) * mesh.vertices.size();
        VkDeviceSize iSize = sizeof(uint32_t) * mesh.indices.size();

        if (!createBuffer(vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                          vertexBuffer)) {
            return false;
        }
        std::memcpy(vertexBuffer.mapped, mesh.vertices.data(), vSize);

        if (!createBuffer(iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                          indexBuffer)) {
            return false;
        }
        std::memcpy(indexBuffer.mapped, mesh.indices.data(), iSize);

        indexCount = static_cast<uint32_t>(mesh.indices.size());
        meshDirty = false;
        return true;
    }

    void updateAnimation(float dt) {
        if (!animate) {
            return;
        }
        animTime += dt * trajectorySpeed;
        const float t = animTime;

        // Complex trajectory computations (Task 3)
        switch (trajectoryType) {
            case 0: { // Lissajous 3D curve
                position.x = trajectoryRadius * std::sin(t);
                position.y = trajectoryRadius * 0.7f * std::sin(2.0f * t);
                position.z = trajectoryHeight * std::cos(3.0f * t);
                break;
            }
            case 1: { // Trefoil Knot (Трилистник)
                position.x = (trajectoryRadius / 3.0f) * (std::sin(t) + 2.0f * std::sin(2.0f * t));
                position.y = (trajectoryRadius / 3.0f) * (std::cos(t) - 2.0f * std::cos(2.0f * t));
                position.z = trajectoryHeight * 0.6f * (-std::sin(3.0f * t));
                break;
            }
            case 2: { // Spatial Spiral (Helix)
                position.x = trajectoryRadius * std::cos(t);
                position.y = trajectoryRadius * std::sin(t);
                position.z = trajectoryHeight * std::sin(0.5f * t);
                break;
            }
            case 3: { // Lemniscate of Bernoulli 3D (Пространственная восьмерка)
                float denom = 1.0f + std::sin(t) * std::sin(t);
                position.x = trajectoryRadius * (std::cos(t) / denom);
                position.y = trajectoryRadius * ((std::sin(t) * std::cos(t)) / denom);
                position.z = trajectoryHeight * std::sin(2.0f * t);
                break;
            }
            default:
                break;
        }

        if (autoRotate) {
            rotation += rotationSpeed * dt;
            rotation.x = std::fmod(rotation.x, 360.0f);
            rotation.y = std::fmod(rotation.y, 360.0f);
            rotation.z = std::fmod(rotation.z, 360.0f);
        }
    }

    glm::mat4 computeModelMatrix() const {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::rotate(model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, scale);
        return model;
    }

    void cleanup() {
        destroyBuffer(vertexBuffer);
        destroyBuffer(indexBuffer);
        destroyBuffer(uniformBuffer);
    }
};

// Global application state
std::vector<TorusObject> objects;
int selectedObjectIndex = 0;

// Camera & Projection settings (Task 1)
enum class ProjectionType {
    Perspective = 0,
    Orthographic = 1
};
ProjectionType currentProjection = ProjectionType::Perspective;

float cameraDistance = 8.0f;
float cameraAzimuth = 45.0f;    // Yaw in degrees
float cameraElevation = 25.0f;  // Pitch in degrees
glm::vec3 cameraTarget{0.0f, 0.0f, 0.0f};

// Perspective parameters
float fovDegrees = 50.0f;
float zNearPersp = 0.1f;
float zFarPersp = 100.0f;

// Orthographic parameters
float orthoSize = 4.5f;
float zNearOrtho = -20.0f;
float zFarOrtho = 50.0f;

// Global animation master control
bool globalAnimationPlaying = true;

// Lighting parameters
glm::vec3 lightDirection{0.577f, 0.707f, 0.408f};
float ambientLight = 0.25f;

// Vulkan Pipeline resources
VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline graphicsPipeline = VK_NULL_HANDLE;

double lastFrameTime = 0.0;

bool createVulkanPipeline() {
    auto& ctx = graphics::internal::context;

    // 1. Descriptor Set Layout (Binding 0: Uniform Buffer)
    VkDescriptorSetLayoutBinding uboLayoutBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &uboLayoutBinding,
    };

    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout\n";
        return false;
    }

    // 2. Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    if (vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    // 3. Shaders
    auto vertShaderCode = readFile("shaders/shader.vert.spv");
    auto fragShaderCode = readFile("shaders/shader.frag.spv");
    if (vertShaderCode.empty() || fragShaderCode.empty()) {
        std::cerr << "Failed to load SPIR-V shader files\n";
        return false;
    }

    VkShaderModule vertShaderModule = createShaderModule(ctx.device, vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(ctx.device, fragShaderCode);

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
        }
    };

    // 4. Vertex Input
    auto bindingDescription = graphics::Vertex::getBindingDescription();
    auto attributeDescriptions = graphics::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    // 5. Input Assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    // 6. Viewport State (Dynamic)
    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };

    // 7. Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE, // Draw two-sided for torus
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    // 8. Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
    };

    // 9. Depth and Stencil State
    VkPipelineDepthStencilStateCreateInfo depthStencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    // 10. Color Blend
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
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    // 11. Dynamic States (Viewport & Scissor)
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    // 12. Create Graphics Pipeline
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
        .layout = pipelineLayout,
        .renderPass = ctx.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
        vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);
        return false;
    }

    vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
    vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);

    // 13. Descriptor Pool for objects
    uint32_t maxObjects = 16;
    VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = maxObjects,
    };

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = maxObjects,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    return true;
}

bool initObjectDescriptor(TorusObject& obj) {
    auto& ctx = graphics::internal::context;

    // Allocate uniform buffer for this object
    VkDeviceSize uboSize = sizeof(UniformBufferObject);
    if (!createBuffer(uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                      obj.uniformBuffer)) {
        return false;
    }

    // Allocate separate VkDescriptorSet for this object (Task 6)
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &obj.descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set for object " << obj.name << '\n';
        return false;
    }

    // Bind uniform buffer to descriptor set
    VkDescriptorBufferInfo bufferInfo = {
        .buffer = obj.uniformBuffer.buffer,
        .offset = 0,
        .range = sizeof(UniformBufferObject),
    };

    VkWriteDescriptorSet descriptorWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = obj.descriptorSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &bufferInfo,
    };

    vkUpdateDescriptorSets(ctx.device, 1, &descriptorWrite, 0, nullptr);
    return true;
}

} // namespace

bool initialize() {
    if (!createVulkanPipeline()) {
        return false;
    }

    // Object 1: Primary Torus (Основной тор)
    {
        TorusObject obj1;
        obj1.name = "Основной тор";
        obj1.majorRadius = 1.6f;
        obj1.minorRadius = 0.5f;
        obj1.radialSegments = 64;
        obj1.tubularSegments = 32;
        obj1.colorMode = 0; // Local XYZ -> RGB
        obj1.position = glm::vec3(0.0f, 0.0f, 0.0f);
        obj1.rotation = glm::vec3(20.0f, 30.0f, 0.0f);
        obj1.scale = glm::vec3(1.0f);
        obj1.objectColor = glm::vec4(1.0f, 0.85f, 0.3f, 1.0f); // Golden tint
        obj1.animate = false;
        obj1.trajectoryType = 0;
        obj1.trajectorySpeed = 1.0f;
        obj1.trajectoryRadius = 2.5f;
        obj1.trajectoryHeight = 1.2f;

        if (!obj1.buildMesh()) {
            return false;
        }
        if (!initObjectDescriptor(obj1)) {
            return false;
        }
        objects.push_back(std::move(obj1));
    }

    // Object 2: Satellite Torus (Второй объект для выполнения Задания 6 на оценку 5)
    {
        TorusObject obj2;
        obj2.name = "Сателлит (Второй тор)";
        obj2.majorRadius = 0.7f;
        obj2.minorRadius = 0.22f;
        obj2.radialSegments = 48;
        obj2.tubularSegments = 24;
        obj2.colorMode = 1; // Rainbow UV
        obj2.position = glm::vec3(3.2f, 0.5f, 0.0f);
        obj2.rotation = glm::vec3(60.0f, 45.0f, 15.0f);
        obj2.scale = glm::vec3(1.0f);
        obj2.objectColor = glm::vec4(0.35f, 0.8f, 1.0f, 1.0f); // Cyan/Azure tint
        obj2.animate = true; // By default moves on trajectory
        obj2.trajectoryType = 2; // Spatial spiral around main torus
        obj2.trajectorySpeed = 1.4f;
        obj2.trajectoryRadius = 3.6f;
        obj2.trajectoryHeight = 1.8f;

        if (!obj2.buildMesh()) {
            return false;
        }
        if (!initObjectDescriptor(obj2)) {
            return false;
        }
        objects.push_back(std::move(obj2));
    }

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    for (auto& obj : objects) {
        obj.cleanup();
    }
    objects.clear();

    if (graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
}

void update(double time) {
    float dt = (lastFrameTime == 0.0) ? 0.016f : static_cast<float>(time - lastFrameTime);
    lastFrameTime = time;

    // Check if any object mesh needs rebuilding
    for (auto& obj : objects) {
        if (obj.meshDirty) {
            obj.buildMesh();
        }
        if (globalAnimationPlaying) {
            obj.updateAnimation(dt);
        }
    }

    // ------------------------------------------------------------------------
    // Mouse Interaction (Orbit Camera, Object Rotation, Zoom, Pan)
    // ------------------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        // Left Mouse Button: Orbit camera
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            cameraAzimuth -= io.MouseDelta.x * 0.35f;
            cameraElevation += io.MouseDelta.y * 0.35f;
            cameraElevation = std::clamp(cameraElevation, -89.0f, 89.0f);
        }
        // Right Mouse Button: Rotate currently selected object
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right) && selectedObjectIndex < static_cast<int>(objects.size())) {
            auto& curObj = objects[selectedObjectIndex];
            curObj.rotation.y -= io.MouseDelta.x * 0.4f;
            curObj.rotation.x += io.MouseDelta.y * 0.4f;
            curObj.rotation.x = std::fmod(curObj.rotation.x, 360.0f);
            curObj.rotation.y = std::fmod(curObj.rotation.y, 360.0f);
        }
        // Middle Mouse Button: Pan camera target
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
            float radAz = glm::radians(cameraAzimuth);
            glm::vec3 right{std::cos(radAz), 0.0f, -std::sin(radAz)};
            cameraTarget -= right * (io.MouseDelta.x * 0.005f * cameraDistance);
            cameraTarget.y += io.MouseDelta.y * 0.005f * cameraDistance;
        }
        // Mouse Wheel: Zoom
        if (io.MouseWheel != 0.0f) {
            if (currentProjection == ProjectionType::Perspective) {
                cameraDistance = std::clamp(cameraDistance - io.MouseWheel * 0.6f, 1.5f, 40.0f);
            } else {
                orthoSize = std::clamp(orthoSize - io.MouseWheel * 0.3f, 0.5f, 25.0f);
            }
        }
    }

    // ------------------------------------------------------------------------
    // ImGui Controls Panel
    // ------------------------------------------------------------------------
    ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(480, 680), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Лабораторная работа №1 (Вариант 7: Тор)")) {
        ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.5f, 1.0f), "Основы 3D-графики: Оценка 5 (Все 6 заданий)");
        ImGui::Separator();

        // --------------------------------------------------------------------
        // ЗАДАНИЕ 1: Проекция и Камера
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("1. Проекция и Камера (Задание 1)", ImGuiTreeNodeFlags_DefaultOpen)) {
            int proj = static_cast<int>(currentProjection);
            ImGui::Text("Тип проекции:");
            ImGui::SameLine();
            if (ImGui::RadioButton("Перспективная", &proj, 0)) {
                currentProjection = ProjectionType::Perspective;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Ортографическая", &proj, 1)) {
                currentProjection = ProjectionType::Orthographic;
            }

            if (currentProjection == ProjectionType::Perspective) {
                ImGui::SliderFloat("FOV (Угол обзора)", &fovDegrees, 20.0f, 110.0f, "%.1f deg");
                ImGui::SliderFloat("Ближняя плоскость (Near)", &zNearPersp, 0.05f, 5.0f, "%.2f");
                ImGui::SliderFloat("Дальняя плоскость (Far)", &zFarPersp, 10.0f, 200.0f, "%.1f");
            } else {
                ImGui::SliderFloat("Размер области (Ortho Size)", &orthoSize, 1.0f, 15.0f, "%.2f");
                ImGui::SliderFloat("Ближняя плоскость (Near)", &zNearOrtho, -50.0f, 5.0f, "%.1f");
                ImGui::SliderFloat("Дальняя плоскость (Far)", &zFarOrtho, 5.0f, 100.0f, "%.1f");
            }

            ImGui::Separator();
            ImGui::Text("Орбитальная камера:");
            ImGui::SliderFloat("Дистанция камеры", &cameraDistance, 2.0f, 25.0f, "%.1f");
            ImGui::SliderFloat("Азимут (Yaw)", &cameraAzimuth, -180.0f, 180.0f, "%.1f deg");
            ImGui::SliderFloat("Угол возвышения (Pitch)", &cameraElevation, -89.0f, 89.0f, "%.1f deg");
            if (ImGui::Button("Сбросить камеру")) {
                cameraDistance = 8.0f;
                cameraAzimuth = 45.0f;
                cameraElevation = 25.0f;
                cameraTarget = glm::vec3(0.0f);
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Управление мышью (на сцене):");
            ImGui::BulletText("ЛКМ + драг — вращение камеры (Orbit)");
            ImGui::BulletText("ПКМ + драг — вращение активного тора");
            ImGui::BulletText("СКМ + драг — сдвиг камеры (Pan)");
            ImGui::BulletText("Колесико — приближение / отдаление (Zoom)");
        }

        // --------------------------------------------------------------------
        // ЗАДАНИЕ 6: Выбор объекта (Несколько объектов через разные VkDescriptorSet)
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Выбор объекта на сцене (Задание 6 - Оценка 5)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Каждый объект рендерится со своим собственным набором дескрипторов (VkDescriptorSet) и независимыми параметрами.");
            std::vector<const char*> itemNames;
            for (const auto& obj : objects) {
                itemNames.push_back(obj.name.c_str());
            }
            ImGui::Combo("Активный объект", &selectedObjectIndex, itemNames.data(), static_cast<int>(itemNames.size()));

            auto& selObj = objects[selectedObjectIndex];
            ImGui::Checkbox("Отображать объект", &selObj.visible);
        }

        auto& curObj = objects[selectedObjectIndex];

        // --------------------------------------------------------------------
        // ЗАДАНИЕ 2: Трансформации фигуры
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("2. Трансформации фигуры (Задание 2)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Позиция (X, Y, Z)", &curObj.position[0], 0.05f);
            ImGui::SliderFloat3("Поворот (Pitch, Yaw, Roll)", &curObj.rotation[0], -180.0f, 180.0f, "%.1f deg");

            ImGui::Checkbox("Равномерный масштаб", &curObj.uniformScale);
            if (curObj.uniformScale) {
                float uniScale = curObj.scale.x;
                if (ImGui::SliderFloat("Масштаб", &uniScale, 0.1f, 5.0f, "%.2f")) {
                    curObj.scale = glm::vec3(uniScale);
                }
            } else {
                ImGui::SliderFloat3("Растяжение (Sx, Sy, Sz)", &curObj.scale[0], 0.1f, 5.0f, "%.2f");
            }

            if (ImGui::Button("Сбросить трансформацию")) {
                curObj.position = glm::vec3(0.0f);
                curObj.rotation = glm::vec3(0.0f);
                curObj.scale = glm::vec3(1.0f);
            }
        }

        // --------------------------------------------------------------------
        // ЗАДАНИЕ 3: Движение по сложной траектории и анимация
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("3. Анимация по сложной траектории (Задание 3)", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (globalAnimationPlaying) {
                if (ImGui::Button("Пауза анимации (Все)")) {
                    globalAnimationPlaying = false;
                }
            } else {
                if (ImGui::Button("Воспроизведение (Все)")) {
                    globalAnimationPlaying = true;
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Анимация данного объекта", &curObj.animate);

            const char* trajectories[] = {
                "Фигура Лиссажу 3D",
                "Трилистник (Trefoil Knot)",
                "Пространственная спираль (Helix)",
                "Восьмерка Бернулли 3D"
            };
            ImGui::Combo("Траектория", &curObj.trajectoryType, trajectories, IM_ARRAYSIZE(trajectories));

            ImGui::SliderFloat("Скорость движения", &curObj.trajectorySpeed, 0.1f, 5.0f, "%.2fx");
            ImGui::SliderFloat("Радиус траектории", &curObj.trajectoryRadius, 0.5f, 8.0f, "%.2f");
            ImGui::SliderFloat("Высота траектории (Z)", &curObj.trajectoryHeight, 0.0f, 5.0f, "%.2f");

            ImGui::Checkbox("Вращение при движении", &curObj.autoRotate);
            if (curObj.autoRotate) {
                ImGui::SliderFloat3("Скорость вращения (град/с)", &curObj.rotationSpeed[0], -180.0f, 180.0f, "%.1f");
            }
        }

        // --------------------------------------------------------------------
        // ЗАДАНИЯ 4 и 5: Цвет объекта и Процедурные цвета вершин
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("4 & 5. Цвет и Процедурная раскраска", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Задание 4: Общий цвет (ColorEdit):");
            ImGui::ColorEdit3("Цвет объекта", &curObj.objectColor[0]);

            ImGui::Separator();
            ImGui::Text("Задание 5: Процедурный цвет вершин:");
            const char* colorModes[] = {
                "Локальные координаты XYZ -> RGB",
                "Цветовой круг (Rainbow UV)",
                "Векторы нормалей -> RGB",
                "Шахматные полосы (Checkered)"
            };
            if (ImGui::Combo("Режим процедурного цвета", &curObj.colorMode, colorModes, IM_ARRAYSIZE(colorModes))) {
                curObj.meshDirty = true;
            }
            ImGui::TextDisabled("Формула пикселя: ЦветВершины * ЦветОбъекта");
        }

        // --------------------------------------------------------------------
        // Параметры геометрии тора
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Параметры геометрии тора")) {
            if (ImGui::SliderFloat("Главный радиус (R)", &curObj.majorRadius, 0.5f, 4.0f, "%.2f")) {
                curObj.meshDirty = true;
            }
            if (ImGui::SliderFloat("Радиус трубки (r)", &curObj.minorRadius, 0.05f, 2.0f, "%.2f")) {
                curObj.meshDirty = true;
            }
            if (ImGui::SliderInt("Сегменты кольца", &curObj.radialSegments, 8, 128)) {
                curObj.meshDirty = true;
            }
            if (ImGui::SliderInt("Сегменты трубки", &curObj.tubularSegments, 6, 64)) {
                curObj.meshDirty = true;
            }

            ImGui::Text("Количество вершин: %d", (curObj.radialSegments + 1) * (curObj.tubularSegments + 1));
            ImGui::Text("Количество треугольников: %d", curObj.indexCount / 3);
        }

        // --------------------------------------------------------------------
        // Освещение
        // --------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Освещение сцены")) {
            ImGui::SliderFloat3("Направление света", &lightDirection[0], -1.0f, 1.0f);
            if (glm::length(lightDirection) < 0.001f) {
                lightDirection = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            ImGui::SliderFloat("Фоновый свет (Ambient)", &ambientLight, 0.05f, 0.8f, "%.2f");
        }

        ImGui::Separator();
        ImGui::Text("FPS: %.1f (%.2f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
    }
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    // Compute View Matrix
    float radAzimuth = glm::radians(cameraAzimuth);
    float radElevation = glm::radians(cameraElevation);
    glm::vec3 cameraPos{
        cameraDistance * std::cos(radElevation) * std::sin(radAzimuth),
        cameraDistance * std::sin(radElevation),
        cameraDistance * std::cos(radElevation) * std::cos(radAzimuth)
    };
    cameraPos += cameraTarget;
    glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));

    // Compute Projection Matrix (Task 1)
    float aspect = (ctx.swapchain_extent.height > 0)
        ? static_cast<float>(ctx.swapchain_extent.width) / static_cast<float>(ctx.swapchain_extent.height)
        : 1.0f;

    glm::mat4 proj;
    if (currentProjection == ProjectionType::Perspective) {
        proj = glm::perspective(glm::radians(fovDegrees), aspect, zNearPersp, zFarPersp);
    } else {
        proj = glm::ortho(-orthoSize * aspect, orthoSize * aspect, -orthoSize, orthoSize, zNearOrtho, zFarOrtho);
    }
    // Vulkan Y coordinate flip
    proj[1][1] *= -1.0f;

    // Update Uniform Buffers for each object before recording commands
    for (auto& obj : objects) {
        if (!obj.visible || obj.uniformBuffer.mapped == nullptr) {
            continue;
        }

        UniformBufferObject ubo{};
        ubo.model = obj.computeModelMatrix();
        ubo.view = view;
        ubo.proj = proj;
        ubo.objectColor = obj.objectColor;
        ubo.lightDir = glm::vec4(glm::normalize(lightDirection), 0.0f);
        ubo.ambientStrength = ambientLight;
        ubo.specularStrength = 0.5f;
        ubo.shininess = 32.0f;

        std::memcpy(obj.uniformBuffer.mapped, &ubo, sizeof(ubo));
    }

    // ------------------------------------------------------------------------
    // Record Command Buffer
    // ------------------------------------------------------------------------
    vkResetCommandBuffer(fd.command_buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.10f, 0.11f, 0.14f, 1.0f}};
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

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

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

    // Render each object (Task 6: Each object binds its own separate VkDescriptorSet)
    for (const auto& obj : objects) {
        if (!obj.visible || obj.indexCount == 0) {
            continue;
        }

        VkBuffer vertexBuffers[] = {obj.vertexBuffer.buffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, obj.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Bind this object's specific descriptor set
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout, 0, 1, &obj.descriptorSet, 0, nullptr);

        vkCmdDrawIndexed(fd.command_buffer, obj.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(fd.command_buffer);

    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
