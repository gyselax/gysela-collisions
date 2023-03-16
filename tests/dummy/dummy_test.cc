#include <KOLIOP/koliop.h>

#include <iostream>

int main(int, char**) {
    if(::koliop_Initialize(8, 0, KOLIOP_TRUE) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
        return -1;
    }

    std::cout << "Initialized()\n";

    ::koliop_Operator an_operator;

    static constexpr uint64_t Npolmax                = 3;
    static constexpr uint64_t index_max              = 4;
    static constexpr uint64_t Nmu                    = 47;
    static constexpr uint64_t Nvpar                  = 127;
    static constexpr uint64_t Nspecies               = 1;
    static constexpr uint64_t pSpecies_id            = 0;
    static constexpr uint64_t mu_id                  = 4; // [0, nproc)
    static constexpr uint64_t istart                 = 0;
    static constexpr uint64_t iend                   = 31;
    static constexpr uint64_t jstart                 = 0;
    static constexpr uint64_t jend                   = 31;
    static constexpr int64_t  collstart              = 0; // [0, nproc)
    static constexpr int64_t  collend                = 0; // [0, nproc)
    static constexpr double   R0                     = 320.0;
    static constexpr uint32_t collision_interspecies = true;
    static constexpr uint32_t SOL                    = false;
    static constexpr uint32_t LIM                    = false;

    // lbound(comb_mat,1) = 0;
    // ubound(comb_mat,1) = 5;
    // size(comb_mat,1) = 6;
    // lbound(comb_mat,2) = 0;
    // ubound(comb_mat,2) = 5;
    // size(comb_mat,2) = 6;
    const double comb_mat[6][6]    = {{1., 0., 0., 0., 0., 0.},
                                      {1.0, 1.0, 4.0, 4.0, 4.0, 4.0},
                                      {1., 2., 1., 0., 0., 0.},
                                      {1., 3., 3., 1., 0., 0.},
                                      {1., 4., 6., 4., 1., 0.},
                                      {1., 5., 10., 10., 5., 1.}};
    uint64_t     comb_mat_extent_0 = 6;
    uint64_t     comb_mat_extent_1 = 6;

    // lbound(hat_As,1) = 0;
    // ubound(hat_As,1) = 2;
    // size(hat_As,1) = 3;
    const double hat_As[]        = {1.0, 0.0, 0.0};
    uint64_t     hat_As_extent_0 = 3;

    // lbound(hat_Zs,1) = 0;
    // ubound(hat_Zs,1) = 2;
    // size(hat_Zs,1) = 3;
    const double hat_Zs[]        = {1.0, 0.0, 0.0};
    uint64_t     hat_Zs_extent_0 = 3;

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
                       &comb_mat[0][0], comb_mat_extent_0, comb_mat_extent_1,
                       hat_As, hat_As_extent_0,
                       hat_Zs, hat_Zs_extent_0) != KOLIOP_STATUS_SUCCESS) {
        std::cout << "::koliop_Initialize() failed\n";
    }

    std::cout << an_operator << "\n";

    const double rg[]            = {1.1111111111111109, 3.333333333333333, 5.5555555555555554, 7.7777777777777777, 10., 12.222222222222223, 14.444444444444445, 16.666666666666668, 18.888888888888889, 21.111111111111111, 23.333333333333336, 25.555555555555557, 27.777777777777779, 30., 32.222222222222221, 34.444444444444443, 36.666666666666671, 38.888888888888893, 41.111111111111114, 43.333333333333336, 45.555555555555557, 47.777777777777779, 50., 52.222222222222221, 54.44444444444445, 56.666666666666671, 58.888888888888893, 61.111111111111114, 63.333333333333336, 65.555555555555557, 67.777777777777786, 70.};
    const double nustaR0_r[]     = {184.14192532377947, 35.438257627946825, 16.470573095840162, 9.9436384303559215, 6.8216922873912118, 5.0499247210943716, 3.9323757140844435, 3.1749322066681667, 2.6341170601726076, 2.2324656348072947, 1.9248612708562103, 1.683433142690395, 1.4901327942584295, 1.3328121864064699, 1.2030268309153078, 1.0947441177054502, 1.0035521961829219, 0.92615821431536416, 0.86006031915208092, 0.80332750744392956, 0.7544483832704707, 0.71222508278551266, 0.67569748768476301, 0.64408816936750446, 0.61676178578755858, 0.59319472388759353, 0.57295211649546463, 0.55567024149265165, 0.54104289974809217, 0.52881076911912361, 0.51875300886823839, 0.51068058303590491};
    const double q_r[]           = {1.2000000674762548, 1.200004387445524, 1.2000305658857675, 1.2001097800745295, 1.2002852807746429, 1.2006115657747147, 1.2011538137443118, 1.2019874570570905, 1.2031978410301916, 1.2048799419166174, 1.2071381273067248, 1.2100859484960278, 1.213845957754154, 1.2185495455007442, 1.224336793733066, 1.231356342953307, 1.2397652704742095, 1.2497289784350696, 1.2614210901940766, 1.2750233540142315, 1.2907255531525996, 1.3087254216125657, 1.3292285649371665, 1.3524483855162783, 1.3786060119570753, 1.4079302321297891, 1.4406574295524956, 1.4770315228216333, 1.517303907831018, 1.5617334025525744, 1.610586194177877, 1.6641357884417383};
    const double mask_buffer_r[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2.81000711525697389E-3, 9.85588964179064941E-2, 0.80923908090803054, 0.99396116715554794};

    // 2*2.77509396353926353E-4,  2.77509396353926407E-4,  2.77509396353926353E-4,  2.77509396353926299E-4,  5*2.77509396353926353E-4,  2.77509396353926299E-4,  2.77509396353926353E-4,  3*2.77509396353926299E-4,  2*2.77509396353926353E-4,  2.77509396353926299E-4,  2.77509396353926353E-4,  2.77509396353926407E-4,  2.77509396353926299E-4,  3*2.77509396353926244E-4,  2.77509396353926299E-4,  2.77509396353926407E-4,  2.77509396353926299E-4,  2.77509396353926407E-4,  2.76729592975621106E-4,  2.50158376503683969E-4,  5.29379475051326199E-5,  1.67583285734615439E-6
    double nustar_coeff[1024];

    ::koliop_PrepareNuStar(an_operator,
                           rg,
                           nustaR0_r,
                           q_r,
                           mask_buffer_r,
                           nustar_coeff);

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
