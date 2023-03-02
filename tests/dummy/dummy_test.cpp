#include <KOLIOP/koliop.h>

#include <iostream>

int main(int, char**) {
    if(::koliop_Initialize(8, 0, 1) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
    } else if (::koliop_Deinitialize() != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Deinitialize() failed\n";
    }
}
