#include "TriangleApp.h"
#include <iostream>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>

// --- constants ---

const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// --- constructor ---

TriangleApp::TriangleApp()
    : _window(WIDTH, HEIGHT, "Vulkan — FPS Camera")
    , _context(enableValidationLayers)
    , _renderer(_context, MAX_FRAMES_IN_FLIGHT)
    , _camera(glm::vec3(0.0f, 0.5f, 3.0f), -90.0f, -10.0f)
{
}

// --- Run ---

void TriangleApp::Run() {
    _context.CreateInstance();
    VkSurfaceKHR surface = _window.CreateSurface(_context.GetInstance());
    _context.Initialize(surface);

    int width, height;
    _window.GetFramebufferSize(&width, &height);
    _renderer.Initialize(width, height);

    _window.SetFramebufferResizeCallback([this](int w, int h) {
        OnFramebufferResize(w, h);
        });
    SetupInputCallbacks();

    glfwSetInputMode(_window.GetHandle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    MainLoop();
    _renderer.WaitIdle();
}

// --- SetupInputCallbacks ---
// GLFW C callbacks can't capture state. We store 'this' as the user pointer
// and retrieve it in each lambda. This overwrites Window's own user pointer,
// so the framebuffer resize callback is re-registered here as well.

void TriangleApp::SetupInputCallbacks() {
    GLFWwindow* win = _window.GetHandle();

    glfwSetWindowUserPointer(win, this);

    glfwSetCursorPosCallback(win, [](GLFWwindow* w, double xpos, double ypos) {
        auto* app = reinterpret_cast<TriangleApp*>(glfwGetWindowUserPointer(w));
        if (app) app->OnMouseMove(xpos, ypos);
        });

    glfwSetKeyCallback(win, [](GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/) {
        auto* app = reinterpret_cast<TriangleApp*>(glfwGetWindowUserPointer(w));
        if (app) app->OnKey(key, action);
        });

    // re-register after overwriting Window's user pointer
    glfwSetFramebufferSizeCallback(win, [](GLFWwindow* w, int width, int height) {
        auto* app = reinterpret_cast<TriangleApp*>(glfwGetWindowUserPointer(w));
        if (app) app->OnFramebufferResize(width, height);
        });
}

// --- MainLoop ---

void TriangleApp::MainLoop() {
    while (!_window.ShouldClose()) {
        _window.PollEvents();

        double currentTime = glfwGetTime();
        float  deltaTime = static_cast<float>(currentTime - _lastFrameTime);
        _lastFrameTime = currentTime;

        ProcessInput(deltaTime);

        float aspect = _renderer.GetAspectRatio();
        _renderer.SetViewMatrix(_camera.GetViewMatrix());
        _renderer.SetProjectionMatrix(_camera.GetProjectionMatrix(aspect));

        _renderer.DrawFrame();

        if (_renderer.FramebufferWasResized()) {
            int w, h;
            _window.GetFramebufferSize(&w, &h);
            while (w == 0 || h == 0) {
                _window.GetFramebufferSize(&w, &h);
                _window.WaitEvents();
            }
            _renderer.RecreateSwapChain(w, h);
        }
    }
}

// --- ProcessInput ---
// polled each frame — key callbacks only fire on press/release, not while held

void TriangleApp::ProcessInput(float deltaTime) {
    GLFWwindow* win = _window.GetHandle();

    if (!_mouseCapture) return;

    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
        _camera.ProcessKeyboard(Camera::FORWARD, deltaTime);
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
        _camera.ProcessKeyboard(Camera::BACKWARD, deltaTime);
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
        _camera.ProcessKeyboard(Camera::LEFT, deltaTime);
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
        _camera.ProcessKeyboard(Camera::RIGHT, deltaTime);
    if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS)
        _camera.ProcessKeyboard(Camera::UP_DIR, deltaTime);
    if (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        _camera.ProcessKeyboard(Camera::DOWN_DIR, deltaTime);
}

// --- OnMouseMove ---
// _firstMouse prevents a large jump on the first event when cursor position is unknown

void TriangleApp::OnMouseMove(double xpos, double ypos) {
    if (!_mouseCapture) return;

    if (_firstMouse) {
        _lastMouseX = xpos;
        _lastMouseY = ypos;
        _firstMouse = false;
        return;
    }

    float xOffset = static_cast<float>(xpos - _lastMouseX);
    float yOffset = static_cast<float>(ypos - _lastMouseY);
    _lastMouseX = xpos;
    _lastMouseY = ypos;

    _camera.ProcessMouseMovement(xOffset, yOffset);
}

// --- OnKey ---
// one-shot actions only — held keys are handled in ProcessInput

void TriangleApp::OnKey(int key, int action) {
    if (action != GLFW_PRESS) return;

    GLFWwindow* win = _window.GetHandle();

    if (key == GLFW_KEY_ESCAPE) {
        if (_mouseCapture) {
            _mouseCapture = false;
            _firstMouse = true; // reset to avoid a delta spike on recapture
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else {
            _mouseCapture = true;
            _firstMouse = true;
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
}

// --- OnFramebufferResize ---

void TriangleApp::OnFramebufferResize(int /*width*/, int /*height*/) {
    _renderer.SetFramebufferResized(true);
}

// --- BenchmarkRun ---
// the model matrix is frozen at identity: rotation would add per-frame UBO writes
// that vary with wall-clock time, introducing noise unrelated to CB recording cost.
// glfwPollEvents is passed to BenchmarkRunner so the OS doesn't kill the window.

void TriangleApp::BenchmarkRun() {
    _context.CreateInstance();
    VkSurfaceKHR surface = _window.CreateSurface(_context.GetInstance());
    _context.Initialize(surface);

    int width, height;
    _window.GetFramebufferSize(&width, &height);
    // must be set before Initialize() so CreateSwapChain picks IMMEDIATE present mode
    _renderer.SetBenchmarkMode(true);
    _renderer.Initialize(width, height);

    // fixed view: no camera movement during benchmark
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.5f, 3.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));
    float aspect = _renderer.GetAspectRatio();
    _renderer.SetViewMatrix(view);
    _renderer.SetProjectionMatrix(_camera.GetProjectionMatrix(aspect));
    _renderer.SetModelAngle(0.0f);

    auto pollFn = [this]() { _window.PollEvents(); };
    BenchmarkRunner runner(_renderer, pollFn);

    BenchmarkResult st = runner.Run(false);
    BenchmarkResult mt = runner.Run(true);

    _renderer.WaitIdle();

    BenchmarkRunner::PrintComparison(st, mt);
    MemoryTracker::Get().PrintReport();
}