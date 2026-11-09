#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace graphics {

struct Texture {
    std::string name;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    int32_t width = 0;
    int32_t height = 0;

    bool isValid() const {
        return imageView != VK_NULL_HANDLE && sampler != VK_NULL_HANDLE;
    }
};

class TextureManager {
public:
    static bool initialize();
    static void shutdown();

    static bool loadFromFile(const std::string& filepath, const std::string& name, Texture& outTexture);
    static bool createSolidColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a, const std::string& name, Texture& outTexture);
    static void destroyTexture(Texture& tex);

    // Fallback 1x1 textures (Task 3 for Grade 5)
    static Texture blackFallback;
    static Texture whiteFallback;
};

} // namespace graphics
