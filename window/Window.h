#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <functional>

class Window {
public:
    Window(int width, int height, const char* title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    VkSurfaceKHR CreateSurface(VkInstance instance);
    void PollEvents();
    bool ShouldClose() const;
    void GetFramebufferSize(int* width, int* height) const;
    void WaitEvents();

    void SetFramebufferResizeCallback(std::function<void(int, int)> callback);

    GLFWwindow* GetHandle() const { return _window; }

private:
    GLFWwindow* _window = nullptr;
    int _width;
    int _height;

    std::function<void(int, int)> _resizeCallback;

    static void FramebufferResizeCallbackStatic(GLFWwindow* window, int width, int height);
};