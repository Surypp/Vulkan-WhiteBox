#include "Renderer.h"
#include "../vulkan/VulkanContext.h"
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <cstdio>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// --- cube geometry ---
// texture atlas 192x64: [0..1/3] side | [1/3..2/3] top | [2/3..1] bottom

static constexpr float U_SIDE_L = 0.0f, U_SIDE_R = 1.0f / 3.0f;
static constexpr float U_TOP_L = 1.0f / 3.0f, U_TOP_R = 2.0f / 3.0f;
static constexpr float U_BOT_L = 2.0f / 3.0f, U_BOT_R = 1.0f;

const std::vector<Vertex> vertices = {
    // +X
    {{ 0.5f,-0.5f,-0.5f},{U_SIDE_L,1.f},{ 1,0,0}}, {{ 0.5f, 0.5f,-0.5f},{U_SIDE_L,0.f},{ 1,0,0}},
    {{ 0.5f, 0.5f, 0.5f},{U_SIDE_R,0.f},{ 1,0,0}}, {{ 0.5f,-0.5f, 0.5f},{U_SIDE_R,1.f},{ 1,0,0}},
    // -X
    {{-0.5f,-0.5f, 0.5f},{U_SIDE_L,1.f},{-1,0,0}}, {{-0.5f, 0.5f, 0.5f},{U_SIDE_L,0.f},{-1,0,0}},
    {{-0.5f, 0.5f,-0.5f},{U_SIDE_R,0.f},{-1,0,0}}, {{-0.5f,-0.5f,-0.5f},{U_SIDE_R,1.f},{-1,0,0}},
    // +Y top
    {{-0.5f, 0.5f,-0.5f},{U_TOP_L,1.f},{0, 1,0}}, {{-0.5f, 0.5f, 0.5f},{U_TOP_L,0.f},{0, 1,0}},
    {{ 0.5f, 0.5f, 0.5f},{U_TOP_R,0.f},{0, 1,0}}, {{ 0.5f, 0.5f,-0.5f},{U_TOP_R,1.f},{0, 1,0}},
    // -Y bottom
    {{-0.5f,-0.5f, 0.5f},{U_BOT_L,1.f},{0,-1,0}}, {{-0.5f,-0.5f,-0.5f},{U_BOT_L,0.f},{0,-1,0}},
    {{ 0.5f,-0.5f,-0.5f},{U_BOT_R,0.f},{0,-1,0}}, {{ 0.5f,-0.5f, 0.5f},{U_BOT_R,1.f},{0,-1,0}},
    // +Z
    {{-0.5f,-0.5f, 0.5f},{U_SIDE_L,1.f},{0,0, 1}}, {{ 0.5f,-0.5f, 0.5f},{U_SIDE_L,0.f},{0,0, 1}},
    {{ 0.5f, 0.5f, 0.5f},{U_SIDE_R,0.f},{0,0, 1}}, {{-0.5f, 0.5f, 0.5f},{U_SIDE_R,1.f},{0,0, 1}},
    // -Z
    {{ 0.5f,-0.5f,-0.5f},{U_SIDE_L,1.f},{0,0,-1}}, {{-0.5f,-0.5f,-0.5f},{U_SIDE_L,0.f},{0,0,-1}},
    {{-0.5f, 0.5f,-0.5f},{U_SIDE_R,0.f},{0,0,-1}}, {{ 0.5f, 0.5f,-0.5f},{U_SIDE_R,1.f},{0,0,-1}},
};

const std::vector<uint32_t> indices = {
     0, 1, 2,  2, 3, 0,   4, 5, 6,  6, 7, 4,
     8, 9,10, 10,11, 8,  12,13,14, 14,15,12,
    16,17,18, 18,19,16,  20,21,22, 22,23,20,
};

// --- constructor / destructor ---

Renderer::Renderer(VulkanContext& context, int maxFramesInFlight)
    : _context(context), _maxFramesInFlight(maxFramesInFlight)
{
    _appStart = std::chrono::high_resolution_clock::now();
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(_context.GetDevice());

    // workers must be stopped before any Vulkan objects are destroyed
    ShutdownWorkers();

    CleanupSwapChain();

    if (_textureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(_context.GetDevice(), _textureSampler, nullptr);
        _textureSampler = VK_NULL_HANDLE;
    }
    _textureImage = GpuImage{};
    _uniformBuffers.clear();

    if (_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_context.GetDevice(), _descriptorPool, nullptr);
        _descriptorPool = VK_NULL_HANDLE;
    }

    _pipeline.Destroy(_context.GetDevice());
    _vertexBuffer = GpuBuffer{};
    _indexBuffer = GpuBuffer{};

    for (int i = 0; i < _maxFramesInFlight; i++) {
        if (i < (int)_imageAvailableSemaphores.size())
            vkDestroySemaphore(_context.GetDevice(), _imageAvailableSemaphores[i], nullptr);
        if (i < (int)_inFlightFences.size())
            vkDestroyFence(_context.GetDevice(), _inFlightFences[i], nullptr);
    }
    for (auto sem : _renderFinishedSemaphores)
        vkDestroySemaphore(_context.GetDevice(), sem, nullptr);

    if (_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(_context.GetDevice(), _queryPool, nullptr);
        _queryPool = VK_NULL_HANDLE;
    }

    if (_commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_context.GetDevice(), _commandPool, nullptr);
        _commandPool = VK_NULL_HANDLE;
    }

}

// --- Initialize ---

void Renderer::Initialize(int width, int height) {
    CreateSwapChain(width, height);
    CreateImageViews();
    CreateRenderPass();
    _pipeline.BuildDescriptorSetLayout(_context.GetDevice());
    _pipeline.Build(_context.GetDevice(), _renderPass, _swapChainExtent,
        "res/shaders/vert.spv", "res/shaders/frag.spv");
    CreateDepthResources();
    CreateFramebuffers();
    CreateCommandPool();
    CreateTextureResources();
    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateUniformBuffers();
    CreateDescriptorPool();
    CreateDescriptorSets();
    CreateCommandBuffers();
    CreateSyncObjects();
    CreateQueryPool();

    InitWorkers();
}

// --- RecreateSwapChain ---

void Renderer::RecreateSwapChain(int width, int height) {
    vkDeviceWaitIdle(_context.GetDevice());
    CleanupSwapChain();

    CreateSwapChain(width, height);
    CreateImageViews();
    CreateRenderPass();
    _pipeline.Build(_context.GetDevice(), _renderPass, _swapChainExtent,
        "res/shaders/vert.spv", "res/shaders/frag.spv");
    CreateDepthResources();
    CreateFramebuffers();
    CreateCommandBuffers();

    _framebufferResized = false;
}

void Renderer::CleanupSwapChain() {
    _depthImage = GpuImage{};

    for (auto fb : _swapChainFrameBuffers)
        vkDestroyFramebuffer(_context.GetDevice(), fb, nullptr);
    _swapChainFrameBuffers.clear();

    if (!_commandBuffers.empty()) {
        vkFreeCommandBuffers(_context.GetDevice(), _commandPool,
            (uint32_t)_commandBuffers.size(), _commandBuffers.data());
        _commandBuffers.clear();
    }

    _pipeline.DestroyPipelineOnly(_context.GetDevice());

    if (_renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(_context.GetDevice(), _renderPass, nullptr);
        _renderPass = VK_NULL_HANDLE;
    }
    for (auto iv : _swapChainImageViews)
        vkDestroyImageView(_context.GetDevice(), iv, nullptr);
    _swapChainImageViews.clear();

    if (_swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(_context.GetDevice(), _swapChain, nullptr);
        _swapChain = VK_NULL_HANDLE;
    }
}

void Renderer::WaitIdle() { vkDeviceWaitIdle(_context.GetDevice()); }

// --- CreateQueryPool ---
// two timestamp slots per frame-in-flight: [frameIndex*2]=begin, [frameIndex*2+1]=end.
// timestamps bracket the full render pass on the primary CB (both ST and MT paths).
// vkCmdResetQueryPool is required before each use — Vulkan 1.0 has no host-side reset.

void Renderer::CreateQueryPool() {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(_context.GetPhysicalDevice(), &props);

    // timestampValidBits == 0 means the queue family does not support timestamps.
    // this is extremely rare on discrete GPUs but required by spec to check.
    QueueFamilyIndices qfi = _context.FindQueueFamilies(_context.GetPhysicalDevice());
    uint32_t queueFamilyIndex = qfi.graphicsFamily.value();
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(_context.GetPhysicalDevice(), &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProps(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(_context.GetPhysicalDevice(), &queueFamilyCount, queueProps.data());
    if (queueProps[queueFamilyIndex].timestampValidBits == 0)
        throw std::runtime_error("Renderer: graphics queue does not support timestamp queries");

    _timestampPeriod = static_cast<double>(props.limits.timestampPeriod);

    VkQueryPoolCreateInfo ci{};
    ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = static_cast<uint32_t>(2 * _maxFramesInFlight);
    if (vkCreateQueryPool(_context.GetDevice(), &ci, nullptr, &_queryPool) != VK_SUCCESS)
        throw std::runtime_error("Renderer: failed to create timestamp query pool");
}

// --- workers ---

void Renderer::InitWorkers() {
    QueueFamilyIndices qfi = _context.FindQueueFamilies(_context.GetPhysicalDevice());

    for (int i = 0; i < NUM_WORKERS; i++) {
        // VkCommandPool is not thread-safe — one pool per worker, no exceptions
        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.queueFamilyIndex = qfi.graphicsFamily.value();
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(_context.GetDevice(), &pi, nullptr, &_workers[i].commandPool) != VK_SUCCESS)
            throw std::runtime_error("Renderer: failed to create worker command pool");

        _workers[i].secondaryCBs.resize(_maxFramesInFlight);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = _workers[i].commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        ai.commandBufferCount = (uint32_t)_maxFramesInFlight;
        if (vkAllocateCommandBuffers(_context.GetDevice(), &ai, _workers[i].secondaryCBs.data()) != VK_SUCCESS)
            throw std::runtime_error("Renderer: failed to allocate worker secondary CBs");

        _workers[i].thread = std::thread([this, i]() {
            WorkerThread& w = _workers[i];
            while (true) {
                w.state.wait(0, std::memory_order_acquire);  // block until state != 0

                if (w.state.load(std::memory_order_acquire) == 3) break;  // shutdown

                try {
                    w.task();
                    w.error = nullptr;
                } catch (...) {
                    w.error = std::current_exception();
                }

                w.state.store(2, std::memory_order_release);
                w.state.notify_one();

                // wait until main acks via WaitWorker — prevents ABA re-execution
                // if wait(0) were called here instead, a rapid re-dispatch (state→1)
                // before this line would be indistinguishable from "task already done"
                w.state.wait(2, std::memory_order_acquire);
            }
            });
    }

    _workersInitialized = true;
}

void Renderer::ShutdownWorkers() {
    if (!_workersInitialized) return;

    for (int i = 0; i < NUM_WORKERS; i++) {
        _workers[i].state.store(3, std::memory_order_release);  // shutdown signal
        _workers[i].state.notify_one();
        if (_workers[i].thread.joinable())
            _workers[i].thread.join();

        if (_workers[i].commandPool != VK_NULL_HANDLE) {
            // destroying the pool also frees all command buffers allocated from it
            vkDestroyCommandPool(_context.GetDevice(), _workers[i].commandPool, nullptr);
            _workers[i].commandPool = VK_NULL_HANDLE;
            _workers[i].secondaryCBs.clear();
        }
    }

    _workersInitialized = false;
}

void Renderer::DispatchWorker(int workerIdx, std::function<void()> task) {
    WorkerThread& w = _workers[workerIdx];
    w.task = std::move(task);
    w.state.store(1, std::memory_order_release);
    w.state.notify_one();
}

void Renderer::WaitWorker(int workerIdx) {
    WorkerThread& w = _workers[workerIdx];
    w.state.wait(1, std::memory_order_acquire);  // wait until state != 1 (= 2 when done)
    // rethrow any exception the worker caught — propagates to the main thread
    if (w.error) std::rethrow_exception(w.error);
    // reset to idle — unblocks worker's wait(2), allowing it to loop back safely
    w.state.store(0, std::memory_order_release);
    w.state.notify_one();
}

// --- RecordSecondaryCommandBuffer ---
// VkCommandBufferInheritanceInfo must specify the exact render pass and framebuffer
// of the parent primary CB — the GPU needs this context to validate the secondary CB.

void Renderer::RecordSecondaryCommandBuffer(int workerIdx,
    uint32_t imageIndex,
    uint32_t frameIndex,
    uint32_t firstIndex,
    uint32_t indexCount)
{
    WorkerThread& w = _workers[workerIdx];

    // implicit reset via vkBeginCommandBuffer avoids VUID-00045 on explicit vkResetCommandBuffer
    VkCommandBufferInheritanceInfo inherit{};
    inherit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inherit.renderPass = _renderPass;
    inherit.framebuffer = _swapChainFrameBuffers[imageIndex];
    inherit.subpass = 0;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // RENDER_PASS_CONTINUE: this CB executes inside an active render pass
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
        VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    bi.pInheritanceInfo = &inherit;

    if (vkBeginCommandBuffer(w.secondaryCBs[frameIndex], &bi) != VK_SUCCESS)
        throw std::runtime_error("Renderer: worker failed to begin secondary command buffer");

    vkCmdBindPipeline(w.secondaryCBs[frameIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline.Handle());

    VkBuffer     vbs[] = { _vertexBuffer.Handle() };
    VkDeviceSize offs[] = { 0 };
    vkCmdBindVertexBuffers(w.secondaryCBs[frameIndex], 0, 1, vbs, offs);
    vkCmdBindIndexBuffer(w.secondaryCBs[frameIndex], _indexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(w.secondaryCBs[frameIndex],
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        _pipeline.Layout(),
        0, 1, &_descriptorSets[frameIndex], // frameIndex, not imageIndex
        0, nullptr);

    // push constants are not inherited — each secondary CB must set them explicitly regardless of RENDER_PASS_CONTINUE_BIT.
    // both workers push the same matrix; redundant by design, not an optimization opportunity.
    // _modelMatrix is written by UpdateUniformBuffer before DispatchWorker — no data race.
    vkCmdPushConstants(w.secondaryCBs[frameIndex], _pipeline.Layout(),
        VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &_modelMatrix);

    vkCmdDrawIndexed(w.secondaryCBs[frameIndex], indexCount, 1, firstIndex, 0, 0);

    if (vkEndCommandBuffer(w.secondaryCBs[frameIndex]) != VK_SUCCESS)
        throw std::runtime_error("Renderer: worker failed to end secondary command buffer");
}

// --- RecordPrimaryCommandBuffer ---
// imageIndex => _swapChainFrameBuffers (0..swapchain_size-1)
// frameIndex => _descriptorSets / _uniformBuffers (0..MAX_FRAMES-1)

void Renderer::RecordPrimaryCommandBuffer(uint32_t imageIndex, uint32_t frameIndex) {
    VkCommandBuffer cmd = _commandBuffers[frameIndex];

    // implicit reset via vkBeginCommandBuffer is safe here. DrawFrame already waited
    // on inFlightFences[frameIndex], guaranteeing this CB is no longer in use.
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        throw std::runtime_error("Renderer: failed to begin primary command buffer");

    const uint32_t queryBase = frameIndex * 2;

    // reset must happen outside a render pass and before the write; Vulkan 1.0 has no host reset
    vkCmdResetQueryPool(cmd, _queryPool, queryBase, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, _queryPool, queryBase);

    std::array<VkClearValue, 2> clears{};
    clears[0].color = { { 0.05f, 0.05f, 0.05f, 1.0f } };
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpi{};
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpi.renderPass = _renderPass;
    rpi.framebuffer = _swapChainFrameBuffers[imageIndex];
    rpi.renderArea.offset = { 0, 0 };
    rpi.renderArea.extent = _swapChainExtent;
    rpi.clearValueCount = (uint32_t)clears.size();
    rpi.pClearValues = clears.data();

    // SECONDARY_COMMAND_BUFFERS instead of INLINE. commands come from worker-recorded secondary CBs
    vkCmdBeginRenderPass(cmd, &rpi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

    // 36 indices (6 faces × 2 triangles × 3 vertices), split evenly across 2 workers
    const uint32_t half = _indexCount / 2;

    DispatchWorker(0, [this, imageIndex, frameIndex, half]() {
        RecordSecondaryCommandBuffer(0, imageIndex, frameIndex, 0, half);
        });
    DispatchWorker(1, [this, imageIndex, frameIndex, half]() {
        RecordSecondaryCommandBuffer(1, imageIndex, frameIndex, half, _indexCount - half);
        });

    WaitWorker(0);
    WaitWorker(1);

    std::array<VkCommandBuffer, NUM_WORKERS> secondaryCBs = {
        _workers[0].secondaryCBs[frameIndex],
        _workers[1].secondaryCBs[frameIndex]
    };
    vkCmdExecuteCommands(cmd, (uint32_t)secondaryCBs.size(), secondaryCBs.data());

    vkCmdEndRenderPass(cmd);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _queryPool, queryBase + 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("Renderer: failed to end primary command buffer");
}

// --- RecordPrimaryCommandBufferST ---
// ST counterpart to RecordPrimaryCommandBuffer. uses VK_SUBPASS_CONTENTS_INLINE:
// draw commands are embedded directly in the primary CB, no secondary CB dispatch.
// the spec forbids mixing INLINE and SECONDARY_COMMAND_BUFFERS in the same render pass.

void Renderer::RecordPrimaryCommandBufferST(uint32_t imageIndex, uint32_t frameIndex) {
    VkCommandBuffer cmd = _commandBuffers[frameIndex];

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        throw std::runtime_error("Renderer: failed to begin primary command buffer (ST)");

    const uint32_t queryBase = frameIndex * 2;

    // reset must happen outside a render pass and before the write; Vulkan 1.0 has no host reset
    vkCmdResetQueryPool(cmd, _queryPool, queryBase, 2);
    // TOP_OF_PIPE: timestamp written when the GPU starts processing this CB
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, _queryPool, queryBase);

    std::array<VkClearValue, 2> clears{};
    clears[0].color        = { { 0.05f, 0.05f, 0.05f, 1.0f } };
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpi{};
    rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpi.renderPass      = _renderPass;
    rpi.framebuffer     = _swapChainFrameBuffers[imageIndex];
    rpi.renderArea      = { { 0, 0 }, _swapChainExtent };
    rpi.clearValueCount = (uint32_t)clears.size();
    rpi.pClearValues    = clears.data();

    vkCmdBeginRenderPass(cmd, &rpi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline.Handle());

    VkBuffer     vbs[]  = { _vertexBuffer.Handle() };
    VkDeviceSize offs[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
    vkCmdBindIndexBuffer(cmd, _indexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        _pipeline.Layout(),
        0, 1, &_descriptorSets[frameIndex],
        0, nullptr);

    // model matrix bypasses the descriptor set — written directly into the command buffer.
    // no staging, no PCIe transfer, no descriptor lookup on the GPU side.
    vkCmdPushConstants(cmd, _pipeline.Layout(),
        VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &_modelMatrix);

    vkCmdDrawIndexed(cmd, _indexCount, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);

    // BOTTOM_OF_PIPE: timestamp written after all pipeline stages complete for this CB
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _queryPool, queryBase + 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
        throw std::runtime_error("Renderer: failed to end primary command buffer (ST)");
}

// --- DrawFrame ---

void Renderer::DrawFrame() {
    // VUID-00045: CBs indexed by frameIndex. vkWaitForFences(inFlightFences[frameIndex])
    //   guarantees CB[frameIndex] is no longer in use before re-recording.
    // VUID-00067: imageAvailable indexed by frameIndex (fence-protected).
    //   renderFinished indexed by imageIndex — vkAcquireNextImageKHR returning X means
    //   image X is no longer presenting, so renderFinishedSemaphores[X] is unsignaled.
    uint32_t frameIndex = (uint32_t)(_currentFrame % _maxFramesInFlight);

    vkWaitForFences(_context.GetDevice(), 1,
        &_inFlightFences[frameIndex], VK_TRUE, UINT64_MAX);

    // fence above guarantees CB[frameIndex] completed — timestamps are available.
    // guard: slot frameIndex was first written on frame _currentFrame _maxFramesInFlight.
    // reading before that (i.e. the first cycle through all frame slots) hits an
    // uninitialized query and triggers VUID-09401.
    if (_currentFrame >= (uint64_t)_maxFramesInFlight) {
        uint64_t ts[2] = {};
        VkResult qr = vkGetQueryPoolResults(
            _context.GetDevice(), _queryPool,
            frameIndex * 2, 2,
            sizeof(ts), ts, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (qr == VK_SUCCESS)
            _lastGpuTimeMs = static_cast<double>(ts[1] - ts[0]) * _timestampPeriod * 1e-6;
    }

    // semaphore is safe to reuse — fence above guarantees this frame slot is idle
    VkSemaphore acquireSem = _imageAvailableSemaphores[frameIndex];

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(_context.GetDevice(), _swapChain,
        UINT64_MAX, acquireSem, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) { _framebufferResized = true; return; }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    UpdateUniformBuffer(frameIndex);

    // reset after wait, before submit — fence must be unsignaled when passed to vkQueueSubmit
    vkResetFences(_context.GetDevice(), 1, &_inFlightFences[frameIndex]);

    auto recordStart = std::chrono::high_resolution_clock::now();
    if (_multiThreaded)
        RecordPrimaryCommandBuffer(imageIndex, frameIndex);
    else
        RecordPrimaryCommandBufferST(imageIndex, frameIndex);
    auto recordEnd      = std::chrono::high_resolution_clock::now();
    _lastRecordingTimeMs = std::chrono::duration<double, std::milli>(recordEnd - recordStart).count();
    _totalFrameTimeMs += _lastRecordingTimeMs;
    _frameCount++;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // imageIndex is safe here: vkAcquireNextImageKHR guarantees image X is no longer presenting,
    // so its renderFinished semaphore has been consumed and is unsignaled.
    VkSemaphore signalSem = _renderFinishedSemaphores[imageIndex];

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1; si.pWaitSemaphores = &acquireSem;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1; si.pCommandBuffers = &_commandBuffers[frameIndex];
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &signalSem;

    VkResult submitResult = vkQueueSubmit(_context.GetGraphicsQueue(), 1, &si, _inFlightFences[frameIndex]);
    if (submitResult != VK_SUCCESS) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "Failed to submit draw command buffer (VkResult %d)", (int)submitResult);
        throw std::runtime_error(msg);
    }

    VkSwapchainKHR swapChains[] = { _swapChain };
    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;   pi.pWaitSemaphores = &signalSem;
    pi.swapchainCount = 1;   pi.pSwapchains = swapChains;
    pi.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(_context.GetPresentQueue(), &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        _framebufferResized = true;
    else if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to present swapchain image");

    _currentFrame++;
}

// --- metrics ---

void Renderer::PrintMetrics() const {
    std::printf("\n+----------------------------------------------+\n");
    std::printf("|         Renderer -- M4 Frame Metrics          |\n");
    std::printf("+----------------------------------------------+\n");
    std::printf("|  frames rendered          : %-10llu       |\n", (unsigned long long)_frameCount);
    std::printf("|  avg FPS                  : %-10.2f       |\n", GetAvgFPS());
    std::printf("|  Temps CB moyen (CPU)     : %-10.4f ms   |\n", GetAvgFrameTimeMs());
    std::printf("+----------------------------------------------+\n\n");
    MemoryTracker::Get().PrintReport();
}

// --- UpdateUniformBuffer ---

void Renderer::UpdateUniformBuffer(uint32_t frameIndex) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - startTime).count();

    float angle = (_modelAngle >= 0.0f) ? _modelAngle : time * glm::radians(45.0f);

    // model stored as a member so recording paths can push it as a push constant
    _modelMatrix = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 1.0f, 0.0f));

    UniformBufferObject ubo{};
    ubo.view = _viewMatrix;
    ubo.proj = _projMatrix;

    _uniformBuffers[frameIndex].Write(&ubo, sizeof(ubo));
}

// --- CreateCommandBuffers ---
// allocates primary CBs only. Recording happens in RecordPrimaryCommandBuffer() each frame

void Renderer::CreateCommandBuffers() {
    // one CB per frame in flight, not per swapchain image, indexed by frameIndex
    _commandBuffers.resize(_maxFramesInFlight);

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = _commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)_commandBuffers.size();
    if (vkAllocateCommandBuffers(_context.GetDevice(), &ai, _commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");
}

// --- CreateVertexBuffer / CreateIndexBuffer ---

void Renderer::CreateVertexBuffer() {
    VkDeviceSize size = sizeof(vertices[0]) * vertices.size();
    _vertexBuffer = GpuBuffer(
        _context.GetDevice(), _context.GetPhysicalDevice(), size,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    UploadContext ctx(_context.GetDevice(), _commandPool, _context.GetGraphicsQueue());
    _vertexBuffer.UploadViaStagingBuffer(vertices.data(), size, _context.GetPhysicalDevice(), ctx);
}

void Renderer::CreateIndexBuffer() {
    _indexCount = (uint32_t)indices.size();
    VkDeviceSize size = sizeof(indices[0]) * indices.size();
    _indexBuffer = GpuBuffer(
        _context.GetDevice(), _context.GetPhysicalDevice(), size,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    UploadContext ctx(_context.GetDevice(), _commandPool, _context.GetGraphicsQueue());
    _indexBuffer.UploadViaStagingBuffer(indices.data(), size, _context.GetPhysicalDevice(), ctx);
}

// --- swapchain helpers ---

void Renderer::CreateSwapChain(int width, int height) {
    SwapChainSupportDetails support = _context.QuerySwapChainSupport(_context.GetPhysicalDevice());
    VkSurfaceFormatKHR fmt = ChooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR   mode = ChooseSwapPresentMode(support.presentModes);
    VkExtent2D         ext = ChooseSwapExtent(support.capabilities, width, height);

    uint32_t imgCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 &&
        imgCount > support.capabilities.maxImageCount)
        imgCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = _context.GetSurface();
    ci.minImageCount = imgCount;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices qfi = _context.FindQueueFamilies(_context.GetPhysicalDevice());
    uint32_t           qIdx[] = { qfi.graphicsFamily.value(), qfi.presentFamily.value() };
    if (qfi.graphicsFamily != qfi.presentFamily) {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = qIdx;
    }
    else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(_context.GetDevice(), &ci, nullptr, &_swapChain) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swap chain");

    vkGetSwapchainImagesKHR(_context.GetDevice(), _swapChain, &imgCount, nullptr);
    _swapChainImages.resize(imgCount);
    vkGetSwapchainImagesKHR(_context.GetDevice(), _swapChain, &imgCount, _swapChainImages.data());
    _swapChainImageFormat = fmt.format;
    _swapChainExtent = ext;
}

void Renderer::CreateImageViews() {
    _swapChainImageViews.resize(_swapChainImages.size());
    for (size_t i = 0; i < _swapChainImages.size(); i++) {
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = _swapChainImages[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = _swapChainImageFormat;
        vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(_context.GetDevice(), &vi, nullptr, &_swapChainImageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swapchain image view");
    }
}

void Renderer::CreateRenderPass() {
    VkAttachmentDescription color{};
    color.format = _swapChainImageFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkAttachmentDescription depth{};
    depth.format = FindDepthFormat();
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> atts = { color, depth };
    VkRenderPassCreateInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ri.attachmentCount = (uint32_t)atts.size(); ri.pAttachments = atts.data();
    ri.subpassCount = 1;                      ri.pSubpasses = &subpass;
    ri.dependencyCount = 1;                      ri.pDependencies = &dep;

    if (vkCreateRenderPass(_context.GetDevice(), &ri, nullptr, &_renderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create render pass");
}

void Renderer::CreateDepthResources() {
    _depthImage = GpuImage::Create2D(
        _context.GetDevice(), _context.GetPhysicalDevice(),
        _swapChainExtent.width, _swapChainExtent.height,
        FindDepthFormat(), VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);
}

void Renderer::CreateFramebuffers() {
    _swapChainFrameBuffers.resize(_swapChainImageViews.size());
    for (size_t i = 0; i < _swapChainImageViews.size(); i++) {
        std::array<VkImageView, 2> atts = { _swapChainImageViews[i], _depthImage.View() };
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = _renderPass;
        fi.attachmentCount = (uint32_t)atts.size();
        fi.pAttachments = atts.data();
        fi.width = _swapChainExtent.width;
        fi.height = _swapChainExtent.height;
        fi.layers = 1;
        if (vkCreateFramebuffer(_context.GetDevice(), &fi, nullptr, &_swapChainFrameBuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer");
    }
}

void Renderer::CreateCommandPool() {
    QueueFamilyIndices qfi = _context.FindQueueFamilies(_context.GetPhysicalDevice());
    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = qfi.graphicsFamily.value();
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(_context.GetDevice(), &pi, nullptr, &_commandPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
}

void Renderer::CreateTextureResources() {
    const uint32_t W = 64, H = 64;
    const VkDeviceSize imgSize = W * H * 4;

    std::vector<uint8_t> pixels(imgSize);
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            uint8_t* p = pixels.data() + (y * W + x) * 4;
            bool checker = ((x / 8) + (y / 8)) % 2 == 0;
            p[0] = checker ? 34 : 101;
            p[1] = checker ? 139 : 67;
            p[2] = checker ? 34 : 33;
            p[3] = 255;
        }
    }

    GpuBuffer staging(
        _context.GetDevice(), _context.GetPhysicalDevice(), imgSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.Map();
    staging.Write(pixels.data(), imgSize);
    staging.Unmap();

    _textureImage = GpuImage::Create2D(
        _context.GetDevice(), _context.GetPhysicalDevice(),
        W, H, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // batch all three image operations into one GPU submission:
    // 3 separate UploadContext calls = 3 fence waits + 3 queue submits.
    // a single submit collapses this to one PCIe round-trip at init time.
    UploadContext ctx(_context.GetDevice(), _commandPool, _context.GetGraphicsQueue());
    VkCommandBuffer cmd = ctx.Begin();
    _textureImage.TransitionLayout(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    _textureImage.CopyFromBuffer(cmd, staging.Handle(), W, H);
    _textureImage.TransitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ctx.End();

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (vkCreateSampler(_context.GetDevice(), &si, nullptr, &_textureSampler) != VK_SUCCESS)
        throw std::runtime_error("Failed to create texture sampler");
}

void Renderer::CreateUniformBuffers() {
    _uniformBuffers.reserve(_maxFramesInFlight);
    for (int i = 0; i < _maxFramesInFlight; i++) {
        _uniformBuffers.emplace_back(
            _context.GetDevice(), _context.GetPhysicalDevice(),
            sizeof(UniformBufferObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        _uniformBuffers.back().Map();
    }
}

void Renderer::CreateDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         (uint32_t)_maxFramesInFlight };
    sizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)_maxFramesInFlight };

    VkDescriptorPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.poolSizeCount = (uint32_t)sizes.size();
    pi.pPoolSizes = sizes.data();
    pi.maxSets = (uint32_t)_maxFramesInFlight;

    if (vkCreateDescriptorPool(_context.GetDevice(), &pi, nullptr, &_descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool");
}

void Renderer::CreateDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(_maxFramesInFlight, _pipeline.DescriptorSetLayout());

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = _descriptorPool;
    ai.descriptorSetCount = (uint32_t)_maxFramesInFlight;
    ai.pSetLayouts = layouts.data();

    _descriptorSets.resize(_maxFramesInFlight);
    if (vkAllocateDescriptorSets(_context.GetDevice(), &ai, _descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor sets");

    for (int i = 0; i < _maxFramesInFlight; i++) {
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = _uniformBuffers[i].Handle();
        bufInfo.offset = 0;
        bufInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imgInfo.imageView = _textureImage.View();
        imgInfo.sampler = _textureSampler;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = _descriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = _descriptorSets[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &imgInfo;

        vkUpdateDescriptorSets(_context.GetDevice(),
            (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }
}

void Renderer::CreateSyncObjects() {
    // imageAvailable: one per frame in flight. the fence vkWaitForFences(inFlightFences[frameIndex])
    // guarantees the acquire semaphore for that slot is no longer in-flight before reuse.
    _imageAvailableSemaphores.resize(_maxFramesInFlight);

    // renderFinished: one per swapchain image, indexed by imageIndex.
    // when vkAcquireNextImageKHR returns imageIndex X, the presentation engine has finished
    // consuming image X — so renderFinishedSemaphores[X] is guaranteed unsignaled and safe to reuse.
    // indexing by frameIndex instead breaks this guarantee at unlimited frame rates (VUID-00067).
    _renderFinishedSemaphores.resize(_swapChainImages.size());

    _inFlightFences.resize(_maxFramesInFlight);

    VkSemaphoreCreateInfo si{}; si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo     fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < _maxFramesInFlight; i++) {
        if (vkCreateSemaphore(_context.GetDevice(), &si, nullptr, &_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(_context.GetDevice(), &fi, nullptr, &_inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sync objects");
    }
    for (size_t i = 0; i < _swapChainImages.size(); i++) {
        if (vkCreateSemaphore(_context.GetDevice(), &si, nullptr, &_renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render-finished semaphore");
    }
}

// --- format / extent helpers ---

VkSurfaceFormatKHR Renderer::ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& formats) const
{
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
    return formats[0];
}

VkPresentModeKHR Renderer::ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& modes) const
{
    for (const auto& m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Renderer::ChooseSwapExtent(
    const VkSurfaceCapabilitiesKHR& caps, int w, int h) const
{
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;
    VkExtent2D e = { (uint32_t)w, (uint32_t)h };
    e.width = std::clamp(e.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    e.height = std::clamp(e.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return e;
}

VkFormat Renderer::FindDepthFormat() const {
    return FindSupportedFormat(
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkFormat Renderer::FindSupportedFormat(const std::vector<VkFormat>& candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features) const
{
    for (VkFormat fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(_context.GetPhysicalDevice(), fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return fmt;
    }
    throw std::runtime_error("Failed to find supported depth format");
}
