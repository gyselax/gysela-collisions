#include <KOLIOP/koliop.h>

#include <iostream>

int main(int, char**) {
    if(::koliop_Initialize(8, 0, KOLIOP_TRUE) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
        return -1;
    }

    std::cout << "Initialized()\n";

    ::koliop_State a_state;

    if(::koliop_Create(&a_state) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
    }

    std::cout << a_state << "\n";

    if(::koliop_Release(a_state) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Release() failed\n";
    }

    if(::koliop_Deinitialize() != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Deinitialize() failed\n";
        return -1;
    }

    std::cout << "Deinitialized()\n";

    return 0;
}
