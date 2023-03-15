#include <KOLIOP/koliop.h>

#include <iostream>

int main(int, char**) {
    if(::koliop_Initialize(8, 0, KOLIOP_TRUE) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
        return -1;
    }

    std::cout << "Initialized()\n";

    ::koliop_Operator an_operator;

    static constexpr uint64_t  Npolmax                = 3;
    static constexpr uint64_t  index_max              = 4;
    static constexpr uint64_t  Nmu                    = 47;
    static constexpr uint64_t  Nvpar                  = 127;
    static constexpr uint64_t  Nspecies               = 1;
    static constexpr uint64_t  pSpecies_id            = 0;
    static constexpr uint64_t  mu_id                  = 4; // [0, nproc)
    static constexpr uint64_t  istart                 = 0;
    static constexpr uint64_t  iend                   = 31;
    static constexpr uint64_t  jstart                 = 0;
    static constexpr uint64_t  jend                   = 31;
    static constexpr int64_t   collstart              = 0; // [0, nproc)
    static constexpr int64_t   collend                = 0; // [0, nproc)
    static constexpr double    R0                     = 320.0;
    static constexpr uint32_t  collision_interspecies = true;
    static constexpr uint32_t  SOL                    = false;
    static constexpr uint32_t  LIM                    = false;

    double dummy[1024];

    // lbound(comb_mat,1) = 0;
    // ubound(comb_mat,1) = 5;
    // size(comb_mat,1) = 6;
    // lbound(comb_mat,2) = 0;
    // ubound(comb_mat,2) = 5;
    // size(comb_mat,2) = 6;
    const double* comb_mat          = dummy;
    uint64_t      comb_mat_extent_0 = 6;
    uint64_t      comb_mat_extent_1 = 6;

    // lbound(hat_As,1) = 0;
    // ubound(hat_As,1) = 2;
    // size(hat_As,1) = 3;
    const double* hat_As          = dummy;
    uint64_t      hat_As_extent_0 = 3;

    // lbound(hat_Zs,1) = 0;
    // ubound(hat_Zs,1) = 2;
    // size(hat_Zs,1) = 3;
    const double* hat_Zs          = dummy;
    uint64_t      hat_Zs_extent_0 = 3;

    // lbound(mug,1) = 0;
    // ubound(mug,1) = 47;
    // size(mug,1) = 48;
    // const koliop::Flo64* mug = dummy;

    // lbound(vparg,1) = 0;
    // ubound(vparg,1) = 127;
    // size(vparg,1) = 128;
    // const koliop::Flo64* vparg = dummy;

    // lbound(coeff_intdmu,1) = 0;
    // ubound(coeff_intdmu,1) = 47;
    // size(coeff_intdmu,1) = 48;
    // const koliop::Flo64* coeff_intdmu = dummy;

    // lbound(coeff_intdvpar,1) = 0;
    // ubound(coeff_intdvpar,1) = 127;
    // size(coeff_intdvpar,1) = 128;
    // const koliop::Flo64* coeff_intdvpar = dummy;

    if(::koliop_Create(&an_operator,
                       Npolmax,
                       index_max,
                       Nmu,
                       Nvpar,
                       Nspecies,
                       pSpecies_id,
                       mu_id,
                       istart,
                       iend,
                       jstart,
                       jend,
                       collstart,
                       collend,
                       R0,
                       collision_interspecies,
                       SOL,
                       LIM,
                       comb_mat, comb_mat_extent_0, comb_mat_extent_1,
                       hat_As, hat_As_extent_0,
                       hat_Zs, hat_Zs_extent_0) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
    }

    std::cout << an_operator << "\n";

    ::koliop_PrepareNuStar(an_operator);

    if(::koliop_Release(an_operator) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Release() failed\n";
    }

    if(::koliop_Deinitialize() != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Deinitialize() failed\n";
        return -1;
    }

    std::cout << "Deinitialized()\n";

    return 0;
}
