DependencyResolver_Resolve(Kokkos
                           VERSION          "4.1.0...<5.0.0"
                           COMPONENTS       Kokkos::kokkos
                           # Look at the system or if the dependency is already
                           # resolved by someone else. Else fallback to
                           # downloading the product.
                           PROVIDERS        SYSTEM
                                            ARCHIVE
                           PROVIDERS_SOURCE "https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz"
                           )

DependencyResolver_Resolve(KokkosKernels
                           VERSION          "4.1.0...<5.0.0"
                           COMPONENTS       Kokkos::kokkoskernels
                           # Look at the system or if the dependency is already
                           # resolved by someone else. Else fallback to
                           # downloading the product.
                           PROVIDERS        SYSTEM
                                            ARCHIVE
                           PROVIDERS_SOURCE "https://github.com/kokkos/kokkos-kernels/releases/download/4.5.01/kokkos-kernels-4.5.01.tar.gz"
                           )

function(${PROJECT_NAME}_target_add_kokkos target_name)
    target_link_libraries("${target_name}"
                          PRIVATE Kokkos::kokkos)
endfunction()

function(${PROJECT_NAME}_target_add_kokkos_kernels target_name)
    target_link_libraries("${target_name}"
                          PRIVATE Kokkos::kokkoskernels)
endfunction()
