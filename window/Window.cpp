#include "Window.h"
#include <stdexcept>


Window::Window(int width, int height, const char* title)
    : _width(width), _height(height) {

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    _window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, FramebufferResizeCallbackStatic);
}


Window::~Window() {
    if (_window) {
        glfwDestroyWindow(_window);
    }
    glfwTerminate();
}


VkSurfaceKHR Window::CreateSurface(VkInstance instance) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, _window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    return surface;
}


void Window::PollEvents() {
    glfwPollEvents();
}

bool Window::ShouldClose() const {
    return glfwWindowShouldClose(_window);
}

void Window::GetFramebufferSize(int* width, int* height) const {
    glfwGetFramebufferSize(_window, width, height);
}

void Window::WaitEvents() {
    glfwWaitEvents();
}


void Window::SetFramebufferResizeCallback(std::function<void(int, int)> callback) {
    _resizeCallback = callback;
}

void Window::FramebufferResizeCallbackStatic(GLFWwindow* window, int width, int height) {
    auto windowObj = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if (windowObj && windowObj->_resizeCallback) {
        windowObj->_resizeCallback(width, height);
    }
}