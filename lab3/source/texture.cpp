#include "texture.hpp"
#include "graphics_internal.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cstring>
#include <vector>

namespace graphics {

Texture TextureManager::blackFallback;
Texture TextureManager::whiteFallback;

namespace {

VkCommandPool s_transferCommandPool = VK_NULL_HANDLE;

VkCommandPool getTransferCommandPool() {
    auto& ctx = graphics::internal::context;
    if (s_transferCommandPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = ctx.graphics_queue_index,
        };
        if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &s_transferCommandPool) != VK_SUCCESS) {
            std::cerr << "Failed to create transfer command pool\n";
        }
    }
    return s_transferCommandPool;
}

VkCommandBuffer beginSingleTimeCommands() {
    auto& ctx = graphics::internal::context;

    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = getTransferCommandPool(),
        .commandBufferCount = 1,
    };

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    auto& ctx = graphics::internal::context;

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };

    vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphics_queue);

    vkFreeCommandBuffers(ctx.device, getTransferCommandPool(), 1, &commandBuffer);
}

void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(cmd);
}

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };

    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(cmd);
}

VkSampler createDefaultSampler() {
    auto& ctx = graphics::internal::context;

    VkSamplerCreateInfo samplerInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VkSampler sampler = VK_NULL_HANDLE;
    if (vkCreateSampler(ctx.device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create texture sampler\n";
        return VK_NULL_HANDLE;
    }
    return sampler;
}

bool createTextureFromPixels(const void* pixels, uint32_t width, uint32_t height,
                            const std::string& name, Texture& outTexture) {
    auto& ctx = graphics::internal::context;
    VkDeviceSize imageSize = width * height * 4;

    // 1. Host Staging Buffer
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;

    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = imageSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VmaAllocationCreateInfo stagingAllocInfo = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    VmaAllocationInfo resultAllocInfo{};
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &stagingAllocInfo,
                        &stagingBuffer, &stagingAlloc, &resultAllocInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create staging buffer for texture: " << name << '\n';
        return false;
    }
    std::memcpy(resultAllocInfo.pMappedData, pixels, imageSize);

    // 2. GPU Device Local VRAM Image
    VkImageCreateInfo imageInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo imageAllocInfo = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    if (vmaCreateImage(ctx.allocator, &imageInfo, &imageAllocInfo,
                       &outTexture.image, &outTexture.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to allocate VRAM image for texture: " << name << '\n';
        vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);
        return false;
    }

    // 3. Copy via GPU commands: UNDEFINED -> TRANSFER_DST -> SHADER_READ_ONLY
    transitionImageLayout(outTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, outTexture.image, width, height);
    transitionImageLayout(outTexture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);

    // 4. Create Image View
    VkImageViewCreateInfo viewInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = outTexture.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &outTexture.imageView) != VK_SUCCESS) {
        std::cerr << "Failed to create image view for texture: " << name << '\n';
        return false;
    }

    // 5. Create Sampler
    outTexture.sampler = createDefaultSampler();
    outTexture.width = static_cast<int32_t>(width);
    outTexture.height = static_cast<int32_t>(height);
    outTexture.name = name;

    return true;
}

} // namespace

bool TextureManager::initialize() {
    // Create standard fallback textures (Task 3 for Grade 5)
    uint32_t blackPixel = 0xFF000000; // RGBA (0, 0, 0, 255)
    uint32_t whitePixel = 0xFFFFFFFF; // RGBA (255, 255, 255, 255)

    if (!createTextureFromPixels(&blackPixel, 1, 1, "Fallback Black (0,0,0)", blackFallback)) {
        return false;
    }
    if (!createTextureFromPixels(&whitePixel, 1, 1, "Fallback White (1,1,1)", whiteFallback)) {
        return false;
    }
    return true;
}

void TextureManager::shutdown() {
    auto& ctx = graphics::internal::context;
    destroyTexture(blackFallback);
    destroyTexture(whiteFallback);

    if (s_transferCommandPool != VK_NULL_HANDLE && ctx.device != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx.device, s_transferCommandPool, nullptr);
        s_transferCommandPool = VK_NULL_HANDLE;
    }
}

bool TextureManager::loadFromFile(const std::string& filepath, const std::string& name, Texture& outTexture) {
    const std::vector<std::string> candidates = {
        filepath,
        "lab3/" + filepath,
        "../" + filepath,
        "../lab3/" + filepath,
        "../../" + filepath,
        "../../lab3/" + filepath,
        "textures/" + filepath,
        "lab3/textures/" + filepath,
    };

    int texWidth = 0, texHeight = 0, texChannels = 0;
    stbi_uc* pixels = nullptr;
    std::string resolvedPath;

    for (const auto& path : candidates) {
        pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (pixels) {
            resolvedPath = path;
            break;
        }
    }

    if (!pixels) {
        std::cerr << "Failed to load texture image from file: " << filepath << '\n';
        return false;
    }

    bool success = createTextureFromPixels(pixels, static_cast<uint32_t>(texWidth),
                                          static_cast<uint32_t>(texHeight), name, outTexture);
    stbi_image_free(pixels);
    return success;
}

bool TextureManager::createSolidColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                                     const std::string& name, Texture& outTexture) {
    uint8_t pixel[4] = { r, g, b, a };
    return createTextureFromPixels(pixel, 1, 1, name, outTexture);
}

void TextureManager::destroyTexture(Texture& tex) {
    auto& ctx = graphics::internal::context;
    if (ctx.device == VK_NULL_HANDLE) return;

    if (tex.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(ctx.device, tex.sampler, nullptr);
        tex.sampler = VK_NULL_HANDLE;
    }
    if (tex.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, tex.imageView, nullptr);
        tex.imageView = VK_NULL_HANDLE;
    }
    if (tex.image != VK_NULL_HANDLE) {
        vmaDestroyImage(ctx.allocator, tex.image, tex.allocation);
        tex.image = VK_NULL_HANDLE;
        tex.allocation = VK_NULL_HANDLE;
    }
}

} // namespace graphics
