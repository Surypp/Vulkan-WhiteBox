#include "app/TriangleApp.h"
#include <iostream>

int main() {
    try {
        TriangleApp app;
        app.Run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
