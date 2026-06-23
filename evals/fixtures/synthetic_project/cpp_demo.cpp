// cpp_demo.cpp
#include <string>
#include <vector>

namespace anvay::core {

class Engine {
public:
    Engine(int threads) : m_threads(threads) {}
    void run_task(const std::string& name) {}
private:
    int m_threads;
};

template<typename T>
T max_value(T a, T b) {
    return (a > b) ? a : b;
}

} // namespace anvay::core
