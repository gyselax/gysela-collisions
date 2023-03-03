#include <KOLIOP/koliop.h>

#include <iostream>

int main(int, char**) {
    if(::koliop_Initialize(8, 0, KOLIOP_TRUE) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
        return -1;
    }

    std::cout << "koliop_Initialize()\n";

    if(::koliop_Deinitialize() != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Deinitialize() failed\n";
        return -1;
    }

    std::cout << "koliop_Deinitialize()\n";

    return 0;
}
