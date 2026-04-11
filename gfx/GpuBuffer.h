#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <cstring>
#include "UploadContext.h"
#include "MemoryTracker.h"

// --- GpuBuffer ---
// RAII wrapper for VkBuffer + VkDeviceMemory.
// _allocatedSize tracks req.size (not the requested size) to account for GPU alignment.

class GpuBuffer {
public:
    GpuBuffer() = default;

    GpuBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memProps)
        : _device(device), _size(size)
    {
        VkBufferCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &info, nullptr, &_buffer) != VK_SUCCESS)
            throw std::runtime_error("GpuBuffer: vkCreateBuffer failed");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, _buffer, &req);
        _allocatedSize = req.size; // actual size after GPU alignment

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = req.size;
        allocInfo.memoryTypeIndex = FindMemoryType(physicalDevice, req.memoryTypeBits, memProps);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &_memory) != VK_SUCCESS)
            throw std::runtime_error("GpuBuffer: vkAllocateMemory failed");

        MemoryTracker::Get().OnAllocate(_allocatedSize, "GpuBuffer");

        vkBindBufferMemory(device, _buffer, _memory, 0);
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // move transfers ownership without notifying the tracker, the allocation itself doesn't change
    GpuBuffer(GpuBuffer&& o) noexcept
        : _device(o._device), _buffer(o._buffer), _memory(o._memory),
        _mapped(o._mapped), _size(o._size), _allocatedSize(o._allocatedSize)
    {
        o._buffer = VK_NULL_HANDLE;
        o._memory = VK_NULL_HANDLE;
        o._mapped = nullptr;
        o._allocatedSize = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& o) noexcept {
        if (this != &o) {
            Destroy();
            _device = o._device;
            _buffer = o._buffer;
            _memory = o._memory;
            _mapped = o._mapped;
            _size = o._size;
            _allocatedSize = o._allocatedSize;
            o._buffer = VK_NULL_HANDLE;
            o._memory = VK_NULL_HANDLE;
            o._mapped = nullptr;
            o._allocatedSize = 0;
        }
        return *this;
    }

    ~GpuBuffer() { Destroy(); }

    VkBuffer     Handle()        const { return _buffer; }
    VkDeviceSize Size()          const { return _size; }
    VkDeviceSize AllocatedSize() const { return _allocatedSize; }
    void*        Mapped()        const { return _mapped; }
    bool         IsValid()       const { return _buffer != VK_NULL_HANDLE; }

    // persistent map — HOST_VISIBLE only
    void Map() {
        if (_mapped) return;
        vkMapMemory(_device, _memory, 0, _size, 0, &_mapped);
    }

    void Unmap() {
        if (!_mapped) return;
        vkUnmapMemory(_device, _memory);
        _mapped = nullptr;
    }

    void Write(const void* data, VkDeviceSize size, VkDeviceSize offset = 0) {
        if (!_mapped) throw std::runtime_error("GpuBuffer::Write: buffer not mapped");
        memcpy(static_cast<char*>(_mapped) + offset, data, static_cast<size_t>(size));
    }

    void UploadViaStagingBuffer(const void* data, VkDeviceSize size,
        VkPhysicalDevice physDev, UploadContext& ctx)
    {
        // staging buffer is created and destroyed here — alloc + free both tracked
        GpuBuffer staging(_device, physDev, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        staging.Map();
        staging.Write(data, size);
        staging.Unmap();

        VkCommandBuffer cmd = ctx.Begin();
        VkBufferCopy region{ 0, 0, size };
        vkCmdCopyBuffer(cmd, staging._buffer, _buffer, 1, &region);
        ctx.End();
    }

private:
    VkDevice       _device = VK_NULL_HANDLE;
    VkBuffer       _buffer = VK_NULL_HANDLE;
    VkDeviceMemory _memory = VK_NULL_HANDLE;
    void* _mapped = nullptr;
    VkDeviceSize   _size = 0;
    VkDeviceSize   _allocatedSize = 0; // actual allocated size including GPU alignment

    void Destroy() {
        if (_mapped) {
            vkUnmapMemory(_device, _memory);
            _mapped = nullptr;
        }
        if (_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(_device, _buffer, nullptr);
            _buffer = VK_NULL_HANDLE;
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
        throw std::runtime_error("GpuBuffer: no suitable memory type");
    }
};