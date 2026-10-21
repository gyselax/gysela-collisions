// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>

#include "KOLIOP/collisions.h"
#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/portability/cmath.h"
#include "KOLIOP/utility/copy.h"

namespace koliop {
    namespace collisions {
        namespace detail {
            /// See Constant::kDiscardedStreamCount for more detail.
            ///
            void GetFreshStreams(OrdinalType                                      discarded_stream_count,
                                 OrdinalType                                      wanted_stream_count,
                                 std::vector<Constant::DeviceExecutionSpaceType>& execution_space_pool) {
                // This represents the null stream.
                Constant::DeviceExecutionSpaceType default_execution_space{};

                // This represents the streams with weird HW queue mapping. We
                // keep them in a vector to avoid implicitly freeing them (which
                // would defeat the purpose).
                std::vector<Constant::DeviceExecutionSpaceType> unused_streams;
                for(OrdinalType istream = 0; istream < discarded_stream_count; ++istream) {
                    unused_streams.push_back(Kokkos::Experimental::partition_space(default_execution_space, 1)[0]);
                }

                for(OrdinalType istream = 0; istream < wanted_stream_count; ++istream) {
                    // TODO(Etienne M): Read docs on hipStreamCreateWithPriority.
                    execution_space_pool.emplace_back(Kokkos::Experimental::partition_space(default_execution_space, 1)[0]);
                }
            }
        } // namespace detail

        template <typename View2D,
                  typename View1D>
        static inline void
        DoPrepareNuStar(const Operator&  an_operator,
                        View2D&&         nustar_coeff,
                        const View1D&    hat_As,
                        const View1D&    hat_Zs,
                        ComputeFloatType coeff_AD) {

            const auto a_host_visible_nustar_coeff = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), nustar_coeff);

            const auto a_host_visible_hat_As     = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_As);
            const auto a_host_visible_hat_Zs     = Kokkos::create_mirror_view_and_copy(Kokkos::view_alloc(Kokkos::DefaultHostExecutionSpace::memory_space{}, Kokkos::WithoutInitializing), hat_Zs);

            const auto& tor2_extent    = an_operator.tor2_extent_;
            const auto& species_extent = an_operator.species_extent_;

            // TODO(Etienne M): LOOP Obvious loop inter exchange optimization ?
            for(OrdinalType ispeciesB = 0; ispeciesB < species_extent; ++ispeciesB) {
                for(OrdinalType ispeciesA = 0; ispeciesA < species_extent; ++ispeciesA) {
                    const ComputeFloatType hat_Aa = a_host_visible_hat_As(ispeciesA);
                    const ComputeFloatType hat_Za = a_host_visible_hat_Zs(ispeciesA);
                    const ComputeFloatType hat_Ab = a_host_visible_hat_As(ispeciesB);
                    const ComputeFloatType hat_Zb = a_host_visible_hat_Zs(ispeciesB);

                    const ComputeFloatType AD_factor = ((hat_Za * hat_Za) * (hat_Zb * hat_Zb)) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa) *
                                                       (static_cast<ComputeFloatType>(1.0) / hat_Aa +
                                                        static_cast<ComputeFloatType>(1.0) / hat_Ab);

                    a_host_visible_nustar_coeff(ispeciesA, ispeciesB) = coeff_AD * AD_factor;
                }
            }

            koliop::utility::Copy::Shallow(Constant::DeviceExecutionSpaceType{},
                                           a_host_visible_nustar_coeff,
                                           nustar_coeff);
        }


        ////////////////////////////////////////////////////////////////////////
        // Operator methods definition
        ////////////////////////////////////////////////////////////////////////

        Operator::Operator(uint64_t tor1_extent,
                           uint64_t tor2_extent,
                           uint64_t tor3_extent,
                           bool     collision_interspecies,
                           Flo64    nu_ref,
                           // NOTE: in practice, this hat_Xs contains 3 values. The size being hardcoded in global::Nspecies_max = 2
                           const koliop::utility::View::UDL1<const Flo64>& hat_As,
                           const koliop::utility::View::UDL1<const Flo64>& hat_Zs,
                           const koliop::utility::View::UDL1<const Flo64>& mug,
                           const koliop::utility::View::UDL1<const Flo64>& vparg,
                           const koliop::utility::View::UDL1<const Flo64>& coeff_intdmu,
                           const koliop::utility::View::UDL1<const Flo64>& coeff_intdvpar,
                           const koliop::utility::View::UDL2<const Flo64>& B_norm,
                           const koliop::utility::View::UDL4<const Flo64>& Bstar_s)
            // Constant, externally computed values
            : mu_extent_{static_cast<OrdinalType>(mug.extent(0))}
            , vpar_extent_{static_cast<OrdinalType>(vparg.extent(0))}
            , tor1_extent_{static_cast<OrdinalType>(tor1_extent)}
            , tor2_extent_{static_cast<OrdinalType>(tor2_extent)}
            , tor3_extent_{static_cast<OrdinalType>(tor3_extent)}
            , species_extent_{static_cast<OrdinalType>(hat_As.extent(0))}
            , collision_interspecies_{collision_interspecies}
            , coeff_AD_{std::sqrt(2.0) * nu_ref}
            , hat_As_{hat_As}
            , hat_Zs_{hat_Zs}
            , mug_{mug}
            , vparg_{vparg}
            , coeff_intdmu_{coeff_intdmu}
            , coeff_intdvpar_{coeff_intdvpar}
            , B_norm_{B_norm}
            , Bstar_s_{Bstar_s}

            // Constant, internally computed values

            , nustar_coeff_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nustar_coeff_"),
                            static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_)}

            // Variable, internally computed values

            , n_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "n_s_"),
                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , U_par_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_0_"),
                       static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , U_par_s_input_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "U_par_s_1_"),
                             static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , T_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_s_"),
                   static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}
            , nu_ab_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "nu_ab_"),
                     static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_), static_cast<std::size_t>(species_extent_)}
            , q_par_on_nT_s_{Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_par_on_nT_s_"),
                             static_cast<std::size_t>(tor1_extent_), static_cast<std::size_t>(tor2_extent_), static_cast<std::size_t>(tor3_extent_), static_cast<std::size_t>(species_extent_)}

            // Miscellaneous tooling
            , execution_space_pool_{}

            // Miscellaneous numerical tooling
            , cvcd_drivers_scheduler_{} {
            KOLIOP_ASSERT(static_cast<OrdinalType>(hat_As_.extent(0)) == species_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(hat_Zs_.extent(0)) == species_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(mug_.extent(0)) == mu_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(vparg_.extent(0)) == vpar_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(coeff_intdmu_.extent(0)) == mu_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(coeff_intdvpar_.extent(0)) == vpar_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(B_norm_.extent(0)) == tor2_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(B_norm_.extent(1)) == tor1_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(Bstar_s_.extent(0)) == vpar_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(Bstar_s_.extent(1)) == tor2_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(Bstar_s_.extent(2)) == tor1_extent_);
            KOLIOP_ASSERT(static_cast<OrdinalType>(Bstar_s_.extent(3)) == species_extent_);

            if(std::getenv("KOLIOP_DEBUG_ALL") != nullptr ||
               std::getenv("KOLIOP_DEBUG_koliop_Create") != nullptr) {
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                // Assert for powers of t
                std::cout << "[KOLIOP][DEBUG] ((mu_extent - 1) & mu_extent) != 0 : " << (((mu_extent_ - 1) & mu_extent_) != 0) << "\n";
                std::cout << "[KOLIOP][DEBUG] ((vpar_extent - 1) & vpar_extent) != 0 : " << (((vpar_extent_ - 1) & vpar_extent_) != 0) << "\n";
#endif
                std::cout << "[KOLIOP][DEBUG] hat_As.data() == nullptr : " << (hat_As_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] hat_Zs.data() == nullptr : " << (hat_Zs_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] mug.data() == nullptr : " << (mug_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] vparg.data() == nullptr : " << (vparg_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] coeff_intdmu.data() == nullptr : " << (coeff_intdmu_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] coeff_intdvpar.data() == nullptr : " << (coeff_intdvpar_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] B_norm.data() == nullptr : " << (B_norm_.data() == nullptr) << "\n";
                std::cout << "[KOLIOP][DEBUG] Bstar_s.data() == nullptr : " << (Bstar_s_.data() == nullptr) << "\n";
                std::cout << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(mu_extent) : " << static_cast<uint64_t>(mu_extent_) << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(vpar_extent) : " << static_cast<uint64_t>(vpar_extent_) << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(tor1_extent) : " << static_cast<uint64_t>(tor1_extent_) << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(tor2_extent) : " << static_cast<uint64_t>(tor2_extent_) << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(tor3_extent) : " << static_cast<uint64_t>(tor3_extent_) << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<uint64_t>(species_extent) : " << static_cast<uint64_t>(species_extent_) << "\n";
                std::cout << "[KOLIOP][DEBUG] static_cast<bool>(collision_interspecies) : " << static_cast<bool>(collision_interspecies_) << "\n";
                std::cout << "[KOLIOP][DEBUG] hat_As.data() : " << hat_As_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] hat_Zs.data() : " << hat_Zs_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] mug.data() : " << mug_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] vparg.data() : " << vparg_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] coeff_intdmu.data() : " << coeff_intdmu_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] coeff_intdvpar.data() : " << coeff_intdvpar_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] B_norm.data() : " << B_norm_.data() << "\n";
                std::cout << "[KOLIOP][DEBUG] Bstar_s.data() : " << Bstar_s_.data() << "\n";
                std::cout << std::flush;
            }

            if(
#if defined(KOLIOP_UTILITY_FLATTEN_EXPECT_POW2_EXTENTS)
                // Assert for powers of two.
                ((mu_extent_ - 1) & mu_extent_) != 0 ||
                ((vpar_extent_ - 1) & vpar_extent_) != 0 ||
#endif
                hat_As_.data() == nullptr ||
                hat_Zs_.data() == nullptr ||
                mug_.data() == nullptr ||
                vparg_.data() == nullptr ||
                coeff_intdmu_.data() == nullptr ||
                coeff_intdvpar_.data() == nullptr ||
                B_norm_.data() == nullptr ||
                Bstar_s_.data() == nullptr) {
                throw std::runtime_error("KOLIOP_STATUS_INVALID_VALUE");
            }

            // Ensure the copies above are completed.
            // Constant::DeviceExecutionSpaceType{}.fence() would also work.
            Kokkos::fence();

            DoPrepareNuStar(*this,
                            nustar_coeff_,
                            hat_As_,
                            hat_Zs_,
                            coeff_AD_);

            detail::GetFreshStreams(Constant::kDiscardedStreamCount,
                                    Constant::kDisjointExecutionSpaceCount,
                                    execution_space_pool_);

            cvcd_drivers_scheduler_.Plan(*this,
                                         Constant::kPresetBlockExtentCount,
                                         Constant::kPresetBlockExtent);

#if defined(KOLIOP_CONSTANT_MEMORY_USAGE)
            // The memory release is handled thanks to the shared ptr and
            // the kokkos view reference counting.
            cvcd_drivers_scheduler_.AllocateScratchMemory(*this);
#endif
        }

        Operator::~Operator() {
            Kokkos::fence();
        }

        void Operator::Fence(OrdinalType execution_space_index) const {
            KOLIOP_ASSERT(static_cast<std::size_t>(execution_space_index) < execution_space_pool_.size());
            execution_space_pool_[execution_space_index].fence();
        }

        void Operator::FenceAll() const {
            for(auto&& an_execution_space : execution_space_pool_) {
                an_execution_space.fence();
            }

            // TODO(Etienne M): Do we full fence or wait for all the streams ?
            // Fence is a lot cheaper it seems. Though if Gysela executes other
            // GPU kernel while koliop works, this could slow things down.
        }

        void Operator::Collisions(ComputeFloatType                          dt,
                                  const koliop::utility::View::UDL6<Flo64>& f5d_s) {
            KOLIOP_COLLISIONS_CHECK_DISTRIBUTION_EXTENTS(f5d_s);

            // Assume all our streams are empty.

            ComputeMoments(f5d_s);
            PrecomputeNu();

            if((species_extent_ > 1) && collision_interspecies_) {
                ComputeMaxwellianEvolution(dt, f5d_s);
                PrecomputeNu();
            }

            ComputeQparOnNT(f5d_s);

            if(species_extent_ > 1) {
                // Cpar operator
                ComputeCpar(dt, f5d_s);
            }

            // Computes Fa = Fa / FM0a
            ComputeMaxwellianUnshiftedLoc(4, f5d_s);

            // Implied Fence(Constant::kDefaultExecutionSpacePoolIndex);

            // CvCd
            Apply(dt, f5d_s);

            // Implied FenceAll();

            // Computes Fa = Fa * FM0a
            ComputeMaxwellianUnshiftedLoc(3, f5d_s);

            if((species_extent_ > 1) && collision_interspecies_) {
                // Cpar operator
                ComputeCpar(dt, f5d_s);
            }

            // Correction to enforce conservation of density and fluid
            // moments
            ComputeDensityCorrection(f5d_s);

            ComputeUparTCorrection(dt, f5d_s);

            if((species_extent_ > 1) && collision_interspecies_) {
                ComputeMaxwellianEvolution(dt, f5d_s);
                PrecomputeNu();
            }

            Fence(Constant::kDefaultExecutionSpacePoolIndex);
        }
    } // namespace collisions
} // namespace koliop
