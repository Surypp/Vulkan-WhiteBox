#pragma once

#include "../window/Window.h"
#include "../vulkan/VulkanContext.h"
#include "../renderer/Renderer.h"
#include "../scene/Camera.h"
#include "../benchmark/BenchmarkRunner.h"

class TriangleApp {
public:
    TriangleApp();
    ~TriangleApp() = default;

    void Run();

    // runs ST then MT, prints comparison table and memory report.
    // cube rotation is frozen (static model matrix), input callbacks suppressed.
    void BenchmarkRun();

private:
    // destruction order is inverse of declaration order — do not reorder
    Window        _window;
    VulkanContext _context;
    Renderer      _renderer;

    Camera _camera;

    // mouse state for frame-to-frame delta computation
    double _lastMouseX = 0.0;
    double _lastMouseY = 0.0;
    bool   _firstMouse = true;   // true = discard first delta (position unknown)
    bool   _mouseCapture = true; // true = FPS cursor mode active

    double _lastFrameTime = 0.0;

    void MainLoop();
    void OnFramebufferResize(int width, int height);
    void ProcessInput(float deltaTime);
    void OnMouseMove(double xpos, double ypos);
    void OnKey(int key, int action);
    void SetupInputCallbacks();
};