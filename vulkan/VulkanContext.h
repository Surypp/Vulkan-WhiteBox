#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include "../core/QueueFamilyIndices.h"
#include "../core/SwapChainSupportDetails.h"

class VulkanContext {
public:
    VulkanContext(bool enableValidation);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // CreateInstance() must be called before Initialize() as surface doesn't exist yet at construction
    void CreateInstance();
    void Initialize(VkSurfaceKHR surface);

    VkInstance       GetInstance()        const { return _instance; }
    VkDevice         GetDevice()          const { return _device; }
    VkPhysicalDevice GetPhysicalDevice()  const { return _physicalDevice; }
    VkQueue          GetGraphicsQueue()   const { return _graphicsQueue; }
    VkQueue          GetPresentQueue()    const { return _presentQueue; }
    VkSurfaceKHR     GetSurface()         const { return _surface; }

    QueueFamilyIndices      FindQueueFamilies(VkPhysicalDevice device) const;
    unsigned int            FindMemoryType(unsigned int typeFilter, VkMemoryPropertyFlags properties) const;
    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) const;

private:
    VkInstance               _instance = VK_NULL_HANDLE;
    VkPhysicalDevice         _physicalDevice = VK_NULL_HANDLE;
    VkDevice                 _device = VK_NULL_HANDLE;
    VkQueue                  _graphicsQueue = VK_NULL_HANDLE;
    VkQueue                  _presentQueue = VK_NULL_HANDLE;
    VkSurfaceKHR             _surface = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;

    bool _enableValidationLayers;

    void SetupDebugMessenger();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

    bool CheckValidationLayerSupport() const;
    bool IsDeviceSuitable(VkPhysicalDevice device) const;
    bool CheckDeviceExtensionSupport(VkPhysicalDevice device) const;
    std::vector<const char*> GetRequiredExtensions() const;

    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};
