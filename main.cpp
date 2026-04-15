#include "app/TriangleApp.h"
#include <cstdio>
#include <string_view>
#include <exception>

int main(int argc, char** argv) {
    bool benchmark = (argc > 1 && std::string_view(argv[1]) == "--benchmark");
    try {
        TriangleApp app;
        if (benchmark)
            app.BenchmarkRun();
        else
            app.Run();
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
