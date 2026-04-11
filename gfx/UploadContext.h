#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <chrono>
#include "MemoryTracker.h"

// --- UploadContext ---
// one-shot command buffer submission with an explicit fence.
// vkQueueWaitIdle would drain the entire queue; a fence waits only on this specific submit,
// which matters when other GPU work is in flight concurrently.
// Each End() notifies MemoryTracker::MarkUploadComplete() and records submit timing.

class UploadContext {
public:
    UploadContext(VkDevice device, VkCommandPool pool, VkQueue queue)
        : _device(device), _pool(pool), _queue(queue) {
    }

    UploadContext(const UploadContext&) = delete;
    UploadContext& operator=(const UploadContext&) = delete;

    VkCommandBuffer Begin() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = _pool;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(_device, &allocInfo, &_cmd) != VK_SUCCESS)
            throw std::runtime_error("UploadContext: failed to allocate command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(_cmd, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("UploadContext: failed to begin command buffer");

        _submitStart = std::chrono::high_resolution_clock::now();

        return _cmd;
    }

    void End() {
        if (vkEndCommandBuffer(_cmd) != VK_SUCCESS)
            throw std::runtime_error("UploadContext: failed to end command buffer");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // flags = 0: unsignaled — waits for GPU to signal
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(_device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
            throw std::runtime_error("UploadContext: failed to create fence");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &_cmd;

        if (vkQueueSubmit(_queue, 1, &submitInfo, fence) != VK_SUCCESS) {
            vkDestroyFence(_device, fence, nullptr);
            throw std::runtime_error("UploadContext: failed to submit command buffer");
        }

        // 5s timeout — wide enough for debug builds, fails cleanly on GPU hang
        constexpr uint64_t TIMEOUT_NS = 5'000'000'000ULL;
        VkResult result = vkWaitForFences(_device, 1, &fence, VK_TRUE, TIMEOUT_NS);
        if (result == VK_TIMEOUT)
            throw std::runtime_error("UploadContext: GPU timeout (> 5s) during upload");
        if (result != VK_SUCCESS)
            throw std::runtime_error("UploadContext: vkWaitForFences failed");

        auto now = std::chrono::high_resolution_clock::now();
        _lastSubmitDurationMs = std::chrono::duration<double, std::milli>(
            now - _submitStart).count();
        _totalSubmitDurationMs += _lastSubmitDurationMs;
        _submitCount++;

        vkDestroyFence(_device, fence, nullptr);
        vkFreeCommandBuffers(_device, _pool, 1, &_cmd);
        _cmd = VK_NULL_HANDLE;

        MemoryTracker::Get().MarkUploadComplete();
    }

    double LastSubmitDurationMs()  const { return _lastSubmitDurationMs; }
    double TotalSubmitDurationMs() const { return _totalSubmitDurationMs; }
    int    SubmitCount()           const { return _submitCount; }

private:
    VkDevice        _device;
    VkCommandPool   _pool;
    VkQueue         _queue;
    VkCommandBuffer _cmd = VK_NULL_HANDLE;

    std::chrono::high_resolution_clock::time_point _submitStart;
    double _lastSubmitDurationMs = 0.0;
    double _totalSubmitDurationMs = 0.0;
    int    _submitCount = 0;
};