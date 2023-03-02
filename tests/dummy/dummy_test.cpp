#include <KOLIOP/koliop.h>

int main() {
    std::cout << "Is the target the host ? " << mylib::portability::HostDeviceMemoryManagement::IsHostTheTargetDevice() << "\n";
    DoTest<mylib::Flo64>(1024);
}
