#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include "../core/Vertex.h"

// --- Pipeline ---
// owns VkDescriptorSetLayout + VkPipelineLayout + VkPipeline
// render pass is passed to Build() because it belongs to the Renderer (swapchain-dependent)
// BuildDescriptorSetLayout() survives swapchain recreation; Build() doesnt

class Pipeline {
public:
    Pipeline()  = default;
    ~Pipeline() { if (_device != VK_NULL_HANDLE) Destroy(_device); }

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    VkPipeline            Handle()              const { return _pipeline; }
    VkPipelineLayout      Layout()              const { return _pipelineLayout; }
    VkDescriptorSetLayout DescriptorSetLayout() const { return _descriptorSetLayout; }

    // binding 0: UBO (vertex), binding 1: combined image sampler (fragment)
    void BuildDescriptorSetLayout(VkDevice device) {
        _device = device;
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding         = 0;
        uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerBinding{};
        samplerBinding.binding         = 1;
        samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerBinding.descriptorCount = 1;
        samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboBinding, samplerBinding };

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings    = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &_descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Pipeline: failed to create descriptor set layout");
    }

    // viewport is baked at creation time, must be called again on swapchain recreation
    void Build(VkDevice device, VkRenderPass renderPass,
               VkExtent2D swapChainExtent,
               const std::string& vertSpvPath,
               const std::string& fragSpvPath)
    {
        auto vertCode = ReadFile(vertSpvPath);
        auto fragCode = ReadFile(fragSpvPath);
        VkShaderModule vertModule = CreateShaderModule(device, vertCode);
        VkShaderModule fragModule = CreateShaderModule(device, fragCode);

        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName  = "main";

        auto bindingDesc = Vertex::GetBindingDescription();
        auto attrDescs   = Vertex::GetAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount   = 1;
        vertexInput.pVertexBindingDescriptions      = &bindingDesc;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
        vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x = 0; viewport.y = 0;
        viewport.width  = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;

        VkRect2D scissor{ {0, 0}, swapChainExtent };

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
        viewportState.scissorCount  = 1; viewportState.pScissors  = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth   = 1.0f;
        rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts    = &_descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &_pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Pipeline: failed to create pipeline layout");

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = 2;
        pipelineInfo.pStages             = stages;
        pipelineInfo.pVertexInputState   = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = &depthStencil;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.layout              = _pipelineLayout;
        pipelineInfo.renderPass          = renderPass;
        pipelineInfo.subpass             = 0;
        pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_pipeline) != VK_SUCCESS)
            throw std::runtime_error("Pipeline: failed to create graphics pipeline");

        // shader modules are only needed during pipeline creation
        vkDestroyShaderModule(device, fragModule, nullptr);
        vkDestroyShaderModule(device, vertModule, nullptr);
    }

    // destroys pipeline + layout but not descriptorSetLayout. Call before RecreateSwapChain
    void DestroyPipelineOnly(VkDevice device) {
        if (_pipeline       != VK_NULL_HANDLE) { vkDestroyPipeline(device,       _pipeline,       nullptr); _pipeline       = VK_NULL_HANDLE; }
        if (_pipelineLayout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, _pipelineLayout, nullptr); _pipelineLayout = VK_NULL_HANDLE; }
    }

    void Destroy(VkDevice device) {
        DestroyPipelineOnly(device);
        if (_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, _descriptorSetLayout, nullptr);
            _descriptorSetLayout = VK_NULL_HANDLE;
        }
        _device = VK_NULL_HANDLE; // prevents ~Pipeline from double-destroying
    }

private:
    VkDevice              _device              = VK_NULL_HANDLE;
    VkPipeline            _pipeline            = VK_NULL_HANDLE;
    VkPipelineLayout      _pipelineLayout      = VK_NULL_HANDLE;
    VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;

    static std::vector<char> ReadFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Pipeline: cannot open shader: " + path);
        size_t size = (size_t)file.tellg();
        std::vector<char> buf(size);
        file.seekg(0); file.read(buf.data(), (std::streamsize)size);
        return buf;
    }

    static VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code) {
        VkShaderModuleCreateInfo info{};
        info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size();
        info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module;
        if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS)
            throw std::runtime_error("Pipeline: failed to create shader module");
        return module;
    }
};
