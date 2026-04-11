#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>
#include "UploadContext.h"
#include "GpuBuffer.h"
#include "MemoryTracker.h"

// --- GpuImage ---
// RAII wrapper for VkImage + VkDeviceMemory + VkImageView.

class GpuImage {
public:
    GpuImage() = default;

    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;

    GpuImage(GpuImage&& o) noexcept
        : _device(o._device), _image(o._image), _memory(o._memory),
        _view(o._view), _format(o._format), _allocatedSize(o._allocatedSize)
    {
        o._image = VK_NULL_HANDLE;
        o._memory = VK_NULL_HANDLE;
        o._view = VK_NULL_HANDLE;
        o._allocatedSize = 0;
    }

    GpuImage& operator=(GpuImage&& o) noexcept {
        if (this != &o) {
            Destroy();
            _device = o._device;
            _image = o._image;
            _memory = o._memory;
            _view = o._view;
            _format = o._format;
            _allocatedSize = o._allocatedSize;
            o._image = VK_NULL_HANDLE;
            o._memory = VK_NULL_HANDLE;
            o._view = VK_NULL_HANDLE;
            o._allocatedSize = 0;
        }
        return *this;
    }

    ~GpuImage() { Destroy(); }

    static GpuImage Create2D(
        VkDevice device, VkPhysicalDevice physDev,
        uint32_t width, uint32_t height,
        VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags memProps,
        VkImageAspectFlags aspectFlags)
    {
        GpuImage img;
        img._device = device;
        img._format = format;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = { width, height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device, &imageInfo, nullptr, &img._image) != VK_SUCCESS)
            throw std::runtime_error("GpuImage: vkCreateImage failed");

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, img._image, &req);
        img._allocatedSize = req.size;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = req.size;
        allocInfo.memoryTypeIndex = FindMemoryType(physDev, req.memoryTypeBits, memProps);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &img._memory) != VK_SUCCESS)
            throw std::runtime_error("GpuImage: vkAllocateMemory failed");

        MemoryTracker::Get().OnAllocate(img._allocatedSize, "GpuImage");

        vkBindImageMemory(device, img._image, img._memory, 0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = img._image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = { aspectFlags, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &viewInfo, nullptr, &img._view) != VK_SUCCESS)
            throw std::runtime_error("GpuImage: vkCreateImageView failed");

        return img;
    }

    VkImage      Handle()        const { return _image; }
    VkImageView  View()          const { return _view; }
    VkFormat     Format()        const { return _format; }
    VkDeviceSize AllocatedSize() const { return _allocatedSize; }
    bool         IsValid()       const { return _image != VK_NULL_HANDLE; }

    // supported transitions:
    //   VK_IMAGE_LAYOUT_UNDEFINED => TRANSFER_DST_OPTIMAL
    //   TRANSFER_DST_OPTIMAL => SHADER_READ_ONLY_OPTIMAL
    void TransitionLayout(UploadContext& ctx,
        VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkCommandBuffer cmd = ctx.Begin();

        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        if (_format == VK_FORMAT_D32_SFLOAT ||
            _format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
            _format == VK_FORMAT_D24_UNORM_S8_UINT)
            aspect = VK_IMAGE_ASPECT_DEPTH_BIT;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = _image;
        barrier.subresourceRange = { aspect, 0, 1, 0, 1 };

        VkPipelineStageFlags srcStage, dstStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
            newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            throw std::invalid_argument("GpuImage::TransitionLayout: unsupported transition");
        }

        vkCmdPipelineBarrier(cmd, srcStage, dstStage,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        ctx.End();
    }

    void CopyFromBuffer(UploadContext& ctx, VkBuffer srcBuffer,
        uint32_t width, uint32_t height)
    {
        VkCommandBuffer cmd = ctx.Begin();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(cmd, srcBuffer, _image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        ctx.End();
    }

private:
    VkDevice       _device = VK_NULL_HANDLE;
    VkImage        _image = VK_NULL_HANDLE;
    VkDeviceMemory _memory = VK_NULL_HANDLE;
    VkImageView    _view = VK_NULL_HANDLE;
    VkFormat       _format = VK_FORMAT_UNDEFINED;
    VkDeviceSize   _allocatedSize = 0;

    void Destroy() {
        if (_view != VK_NULL_HANDLE) {
            vkDestroyImageView(_device, _view, nullptr);
            _view = VK_NULL_HANDLE;
        }
        if (_image != VK_NULL_HANDLE) {
            vkDestroyImage(_device, _image, nullptr);
            _image = VK_NULL_HANDLE;
        }
        if (_memory != VK_NULL_HANDLE) {
            if (_allocatedSize > 0)
                MemoryTracker::Get().OnFree(_allocatedSize);
            vkFreeMemory(_device, _memory, nullptr);
            _memory = VK_NULL_HANDLE;
            _allocatedSize = 0;
        }
    }

    static uint32_t FindMemoryType(VkPhysicalDevice physDev,
        uint32_t typeFilter,
        VkMemoryPropertyFlags props)
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
            if ((typeFilter & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("GpuImage: no suitable memory type");
    }
};