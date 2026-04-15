#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <array>
#include <thread>
#include <functional>
#include <chrono>
#include <atomic>
#include <exception>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "../core/Vertex.h"
#include "../core/UniformBufferObject.h"
#include "../core/SwapChainSupportDetails.h"
#include "../core/QueueFamilyIndices.h"

#include "../gfx/UploadContext.h"
#include "../gfx/GpuBuffer.h"
#include "../gfx/GpuImage.h"
#include "../gfx/Pipeline.h"
#include "../gfx/MemoryTracker.h"

class VulkanContext;

// --- WorkerThread ---
// one VkCommandPool per worker-pools are not thread-safe.
// per-frame protocol: main dispatches task => worker records secondary CB => main waits => vkCmdExecuteCommands

struct WorkerThread {
    VkCommandPool                commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> secondaryCBs;

    std::thread           thread;
    std::atomic<int>      state{ 0 };  // 0=idle, 1=ready, 2=done, 3=shutdown
    std::function<void()> task;
    std::exception_ptr    error;       // non-null if task threw; checked by WaitWorker

    // non-copyable (atomic is non-movable)
    WorkerThread() = default;
    WorkerThread(const WorkerThread&) = delete;
    WorkerThread& operator=(const WorkerThread&) = delete;
};

// --- Renderer ---

class Renderer {
public:
    Renderer(VulkanContext& context, int maxFramesInFlight = 2);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void Initialize(int width, int height);
    void RecreateSwapChain(int width, int height);
    void DrawFrame();
    void WaitIdle();

    void SetFramebufferResized(bool r) { _framebufferResized = r; }
    bool FramebufferWasResized() const { return _framebufferResized; }

    // toggle between MT (2 workers, secondary CBs) and ST (inline, primary CB only).
    // must be called outside of a frame — not safe to toggle mid-flight.
    void   SetMultiThreaded(bool mt)      { _multiThreaded = mt; }
    double GetLastRecordingTimeMs() const { return _lastRecordingTimeMs; }

    // GPU execution time of the last submitted frame, measured via VkQueryPool timestamps.
    // 0.0 on the first frame (results not yet available when the fence first signals).
    double GetLastGpuTimeMs() const { return _lastGpuTimeMs; }

    // -1.0 = time-based rotation (interactive mode). >= 0 = fixed angle in radians (benchmark mode).
    void   SetModelAngle(float rad)       { _modelAngle = rad; }

    void SetViewMatrix(const glm::mat4& v) { _viewMatrix = v; }
    void SetProjectionMatrix(const glm::mat4& p) { _projMatrix = p; }

    float GetAspectRatio() const {
        if (_swapChainExtent.height == 0) return 1.0f;
        return (float)_swapChainExtent.width / (float)_swapChainExtent.height;
    }

    double GetAvgFrameTimeMs()   const {
        return _frameCount > 0 ? _totalFrameTimeMs / _frameCount : 0.0;
    }
    double GetAvgFPS() const {
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - _appStart).count();
        return _frameCount > 0 ? _frameCount / elapsed : 0.0;
    }
    uint64_t GetFrameCount() const { return _frameCount; }
    void PrintMetrics() const;

private:
    VulkanContext& _context;
    int            _maxFramesInFlight;

    glm::mat4 _viewMatrix = glm::mat4(1.0f);
    glm::mat4 _projMatrix = glm::mat4(1.0f);

    VkSwapchainKHR             _swapChain = VK_NULL_HANDLE;
    std::vector<VkImage>       _swapChainImages;
    VkFormat                   _swapChainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D                 _swapChainExtent = {};
    std::vector<VkImageView>   _swapChainImageViews;
    std::vector<VkFramebuffer> _swapChainFrameBuffers;

    VkRenderPass _renderPass = VK_NULL_HANDLE;

    GpuImage _depthImage;
    GpuImage _textureImage;
    VkSampler _textureSampler = VK_NULL_HANDLE;

    Pipeline _pipeline;

    VkDescriptorPool             _descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> _descriptorSets;

    std::vector<GpuBuffer> _uniformBuffers;

    VkCommandPool                _commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> _commandBuffers; // one per frame in flight, indexed by frameIndex

    GpuBuffer _vertexBuffer;
    GpuBuffer _indexBuffer;
    uint32_t  _indexCount = 0; // stored for the multithreaded split

    // semaphores indexed by frameIndex, not imageIndex — see DrawFrame
    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
    std::vector<VkFence>     _inFlightFences;
    uint64_t                 _currentFrame = 0;
    bool                     _framebufferResized = false;

    static constexpr int NUM_WORKERS = 2;
    std::array<WorkerThread, NUM_WORKERS> _workers;
    bool _workersInitialized = false;

    bool      _multiThreaded       = true;
    double    _lastRecordingTimeMs = 0.0;
    double    _lastGpuTimeMs       = 0.0;
    float     _modelAngle          = -1.0f;
    // member rather than local: UpdateUniformBuffer and Record* are separate calls within DrawFrame —
    // the matrix must survive across that boundary without passing it as a parameter through DispatchWorker lambdas.
    glm::mat4 _modelMatrix = glm::mat4(1.0f);

    VkQueryPool _queryPool       = VK_NULL_HANDLE;
    double      _timestampPeriod = 1.0; // nanoseconds per tick, from VkPhysicalDeviceLimits

    std::chrono::high_resolution_clock::time_point _appStart;
    uint64_t _frameCount = 0;
    double   _totalFrameTimeMs = 0.0; // CB recording time only

    void CreateSwapChain(int width, int height);
    void CreateImageViews();
    void CreateRenderPass();
    void CreateDepthResources();
    void CreateFramebuffers();
    void CreateCommandPool();
    void CreateTextureResources();
    void CreateVertexBuffer();
    void CreateIndexBuffer();
    void CreateUniformBuffers();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void CreateCommandBuffers(); 
    void CreateSyncObjects();
    void CreateQueryPool();

    void InitWorkers();
    void ShutdownWorkers();
    void DispatchWorker(int workerIdx, std::function<void()> task);
    void WaitWorker(int workerIdx);

    // re-recorded every frame; imageIndex indexes framebuffers, frameIndex indexes per-frame resources
    void RecordPrimaryCommandBuffer(uint32_t imageIndex, uint32_t frameIndex);

    // ST path: VK_SUBPASS_CONTENTS_INLINE, no secondary CBs, no worker dispatch.
    // mutually exclusive with RecordPrimaryCommandBuffer per Vulkan spec 7.1.
    void RecordPrimaryCommandBufferST(uint32_t imageIndex, uint32_t frameIndex);

    // recorded by a worker thread; draws [firstIndex, firstIndex+indexCount)
    void RecordSecondaryCommandBuffer(int workerIdx,
        uint32_t imageIndex,
        uint32_t frameIndex,
        uint32_t firstIndex,
        uint32_t indexCount);

    void UpdateUniformBuffer(uint32_t frameIndex);
    void CleanupSwapChain();

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&) const;
    VkPresentModeKHR   ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>&) const;
    VkExtent2D         ChooseSwapExtent(const VkSurfaceCapabilitiesKHR&, int w, int h) const;

    VkFormat FindDepthFormat() const;
    VkFormat FindSupportedFormat(const std::vector<VkFormat>&,
        VkImageTiling, VkFormatFeatureFlags) const;
};