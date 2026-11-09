// SPDX-License-Identifier: MIT

#include "KOLIOP/collisions/cvcddriverscheduler.h"

#include "KOLIOP/collisions.h"
#include "KOLIOP/collisions/cvcddriver.h"
#include "KOLIOP/collisions/cvcddrivereraser.h"
#include "KOLIOP/internal/configuration.h"

namespace koliop {
    namespace collisions {
        namespace detail {
            // NOTE: The amount of memory used internally by the operator
            // depends on the number of disjoint execution space allowing
            // asynchronous blocked CvCd computation and on the maximum block
            // size that can be used to fill the rank local domain.
            // For a fixed kDisjointExecutionSpaceCount lowering the block size
            // wont have an impact on the memory consumption until all the
            // kDisjointExecutionSpaceCount have at least a block each to
            // process. Else, lowering either kDisjointExecutionSpaceCount (down
            // to 1) and lowering the block size can help reduce the memory
            // consumption.
            // The CvCdDriver memory we allocate is:
            //  CvCdDriverVectorSize() * CvCdDriverMatrixSize() * CvCdDriverInstance
            //      Where CvCdDriverInstance = min(kDisjointExecutionSpaceCount,
            //                                     (itor1*itor2*itor3)/(itor1_block*itor2_block*itor3_block))
            //          Where itor*_block are the extent of the largest
            //          kPresetBlockExtent we could use.
            //      For instance, with a domain of (64, 128, 1):
            //          kDisjointExecutionSpaceCount = 2, blocking of (r,theta,phi) = (64, 128, 1)
            //              CvCdDriverInstance = min(2, 8192/8192=1)
            //          kDisjointExecutionSpaceCount = 2, blocking of (r,theta,phi) = (64, 32, 1)
            //              CvCdDriverInstance = min(2, 8192/2048=4)

            /// {Tor1, Tor2, Tor3} block extents.
            /// NOTE: As GPUs get bigger, we also need more parallelism. So
            /// maybe in two years, increase the block size to unlock
            /// Nspace = 32768.
            /// For instance, if an MI300X has 304 CUs. Each CU can take 32
            /// wavefronts (of 64 threads) but we may only be able to fit half.
            /// So 16 * 304 = 4864 waves must be in flight at a given time.
            /// Because we tend to have small workgroups (64 or 128 threads) we
            /// should strive to have bigger CvCd blocks to fill all theses CUs.
            ///
            static inline constexpr OrdinalType kPresetBlockExtent[][3] = {
                // Nspace = 65536
                // {256, 128, 2},
                // {128, 256, 2},
                // Nspace = 32768
                {128, 256, 1},
                {256, 128, 1},
                {128, 128, 2},
                {64, 128, 4},
                {128, 64, 4},
                {32, 128, 8},
                // Nspace = 16384
                {128, 128, 1},
                {64, 128, 2},
                {32, 128, 4},
                {128, 64, 2},
                // Nspace = 8192
                {64, 128, 1},
                {32, 128, 2},
                {128, 64, 1},
                {64, 64, 2},
                {128, 32, 2},
                // Nspace = 4096
                {32, 128, 1},
                {64, 64, 1},
                {32, 64, 2},
                {128, 32, 1},
                {64, 32, 2},
                // Nspace = 2048
                {32, 64, 1},
                {64, 32, 1},
                {32, 32, 2},
                // Nspace = 1024
                {32, 32, 1},
                // Nspace = 512
                {8, 64, 1},
                // Nspace = 16
                {4, 4, 1},
                // Nspace = 1
                // NOTE: Degraded operation performance is expected for 1x1x1
                // blocks.
                {1, 1, 1}};

            static inline constexpr std::size_t kPresetBlockExtentCount = std::size(kPresetBlockExtent);

            template <std::size_t I>
            inline constexpr bool kIsFallbackPreset = kPresetBlockExtent[I][0] == 1 &&
                                                      kPresetBlockExtent[I][1] == 1 &&
                                                      kPresetBlockExtent[I][2] == 1;

            // The driver type for preset I. NOTE: 1x1x1 must use the fake batch traits.
            template <std::size_t I>
            using PresetCvCdDriver = CvCdDriver<kPresetBlockExtent[I][0],
                                                kPresetBlockExtent[I][1],
                                                kPresetBlockExtent[I][2],
                                                std::conditional_t<kIsFallbackPreset<I>,
                                                                   Constant::FakeBatchTraitsType,
                                                                   Constant::BatchTraitsType>>;

            template <std::size_t I, typename... Args>
            bool TryCreateCvCdDriver(CvCdDriverEraser& a_cvcd_driver,
                                     OrdinalType       block_tor1_extent,
                                     OrdinalType       block_tor2_extent,
                                     OrdinalType       block_tor3_extent,
                                     const Args&... args) {
                if(block_tor1_extent != kPresetBlockExtent[I][0] ||
                   block_tor2_extent != kPresetBlockExtent[I][1] ||
                   block_tor3_extent != kPresetBlockExtent[I][2]) {
                    return false;
                }
                a_cvcd_driver.Set(PresetCvCdDriver<I>{args...});
                return true;
            }

            template <typename... Args, std::size_t... Is>
            bool CreateCvCdDriver(std::index_sequence<Is...>,
                                  CvCdDriverEraser& a_cvcd_driver,
                                  OrdinalType       block_tor1_extent,
                                  OrdinalType       block_tor2_extent,
                                  OrdinalType       block_tor3_extent,
                                  const Args&... args) {
                return (TryCreateCvCdDriver<Is>(a_cvcd_driver,
                                                block_tor1_extent, block_tor2_extent, block_tor3_extent,
                                                args...) ||
                        // Short-circuits on the first matching preset.
                        ...);
            }

            // Completes every preset driver type here, in a non-deferred context, before the nvcc stub names
            // Najl<CvCdDriver<...>>. Drop it once Najl no longer takes the driver as template parameter.
            template <std::size_t... Is>
            constexpr bool CompleteAllPresetCvCdDrivers(std::index_sequence<Is...>) {
                return ((sizeof(PresetCvCdDriver<Is>) > 0) && ...);
            }

            // Must be at namespace scope (not inside a function template) so it is evaluated immediately.
            static_assert(CompleteAllPresetCvCdDrivers(std::make_index_sequence<kPresetBlockExtentCount>{}));

            void CreateCvCdDriver(const Operator&   an_operator,
                                  CvCdDriverEraser& a_cvcd_driver,
                                  OrdinalType       block_tor1_extent,
                                  OrdinalType       block_tor2_extent,
                                  OrdinalType       block_tor3_extent) {
                const bool is_created = CreateCvCdDriver(std::make_index_sequence<kPresetBlockExtentCount>{},
                                                         a_cvcd_driver,
                                                         block_tor1_extent, block_tor2_extent, block_tor3_extent,
                                                         an_operator.vpar_extent_,
                                                         an_operator.species_extent_,
                                                         an_operator.collision_interspecies_,
                                                         an_operator.hat_As_,
                                                         an_operator.hat_Zs_,
                                                         an_operator.vparg_,
                                                         an_operator.coeff_intdvpar_);
                KOLIOP_UNUSED(is_created);
                KOLIOP_ASSERT(is_created && "The CvCd operator blocking configuration you chose is not part of kPresetBlockExtent.");
            }

            static inline void
            PrepareBlockPosition(const OrdinalType                                          local_domain_tor1_extent,
                                 const OrdinalType                                          local_domain_tor2_extent,
                                 const OrdinalType                                          local_domain_tor3_extent,
                                 std::vector<CvCdDriverScheduler::CvCdDriverConfiguration>& cvcd_drivers_configurations) {

                if(!((local_domain_tor1_extent > 0) && ((local_domain_tor1_extent & (local_domain_tor1_extent - 1)) == 0) &&
                     (local_domain_tor2_extent > 0) && ((local_domain_tor2_extent & (local_domain_tor2_extent - 1)) == 0) &&
                     (local_domain_tor3_extent > 0) && ((local_domain_tor3_extent & (local_domain_tor3_extent - 1)) == 0))) {
                    throw std::runtime_error(
                        "For performance reason, the number of points in "
                        "each dimension must be a power of 2.");
                }

                auto       first_block_extent = kPresetBlockExtent;
                const auto last_block_extent  = first_block_extent + kPresetBlockExtentCount;

                for(; first_block_extent != last_block_extent; ++first_block_extent) {
                    const auto& current_block_extent = *first_block_extent;

                    const auto current_tor1_block_extent = current_block_extent[0];
                    const auto current_tor2_block_extent = current_block_extent[1];
                    const auto current_tor3_block_extent = current_block_extent[2];

                    KOLIOP_ASSERT(current_tor1_block_extent > 0 &&
                                  current_tor2_block_extent > 0 &&
                                  current_tor3_block_extent > 0);

                    const auto block_count_in_tor1 = local_domain_tor1_extent / current_tor1_block_extent;
                    const auto block_count_in_tor2 = local_domain_tor2_extent / current_tor2_block_extent;
                    const auto block_count_in_tor3 = local_domain_tor3_extent / current_tor3_block_extent;

                    if(block_count_in_tor1 > 0 &&
                       block_count_in_tor2 > 0 &&
                       block_count_in_tor3 > 0) {

                        if(std::getenv("KOLIOP_DEBUG_ALL") != nullptr ||
                           std::getenv("KOLIOP_DEBUG_PrepareBlockPosition") != nullptr) {
                            std::cout << "[KOLIOP][DEBUG] current_tor1_block_extent: " << current_tor1_block_extent << "\n";
                            std::cout << "[KOLIOP][DEBUG] current_tor2_block_extent: " << current_tor2_block_extent << "\n";
                            std::cout << "[KOLIOP][DEBUG] current_tor3_block_extent: " << current_tor3_block_extent << "\n";

                            std::cout << "[KOLIOP][DEBUG] Block count (for each species): " << (block_count_in_tor1 * block_count_in_tor2 * block_count_in_tor3) << "\n";

                            std::cout << std::flush;
                        }

                        OrdinalType block_first_tor3 = local_domain_tor3_extent;
                        for(OrdinalType block_index_in_tor3 = 0; block_index_in_tor3 < block_count_in_tor3; ++block_index_in_tor3) {
                            block_first_tor3 -= current_tor3_block_extent;

                            OrdinalType block_first_tor2 = local_domain_tor2_extent;
                            for(OrdinalType block_index_in_tor2 = 0; block_index_in_tor2 < block_count_in_tor2; ++block_index_in_tor2) {
                                block_first_tor2 -= current_tor2_block_extent;

                                OrdinalType block_first_tor1 = local_domain_tor1_extent;
                                for(OrdinalType block_index_in_tor1 = 0; block_index_in_tor1 < block_count_in_tor1; ++block_index_in_tor1) {
                                    block_first_tor1 -= current_tor1_block_extent;

                                    auto& a_driver_configuration = cvcd_drivers_configurations.emplace_back();

                                    a_driver_configuration.block_first_tor1_ = block_first_tor1;
                                    a_driver_configuration.block_first_tor2_ = block_first_tor2;
                                    a_driver_configuration.block_first_tor3_ = block_first_tor3;

                                    a_driver_configuration.block_tor1_extent_ = current_tor1_block_extent;
                                    a_driver_configuration.block_tor2_extent_ = current_tor2_block_extent;
                                    a_driver_configuration.block_tor3_extent_ = current_tor3_block_extent;

                                    // if(std::getenv("KOLIOP_DEBUG_ALL") != nullptr ||
                                    //    std::getenv("KOLIOP_DEBUG_PrepareBlockPosition") != nullptr) {
                                    //     std::cout << "[KOLIOP][DEBUG] a_driver_configuration.block_first_tor1_: " << a_driver_configuration.block_first_tor1_ << "\n";
                                    //     std::cout << "[KOLIOP][DEBUG] a_driver_configuration.block_first_tor2_: " << a_driver_configuration.block_first_tor2_ << "\n";
                                    //     std::cout << "[KOLIOP][DEBUG] a_driver_configuration.block_first_tor3_: " << a_driver_configuration.block_first_tor3_ << "\n";
                                    //     std::cout << "[KOLIOP][DEBUG] a_driver_configuration.block_tor1_extent_: " << a_driver_configuration.block_tor1_extent_ << "\n";
                                    //     std::cout << "[KOLIOP][DEBUG] a_driver_configuration.block_tor2_extent_: " << a_driver_configuration.block_tor2_extent_ << "\n";
                                    //     std::cout << "[KOLIOP][DEBUG] a_driver_configuration.block_tor3_extent_: " << a_driver_configuration.block_tor3_extent_ << "\n";

                                    //     std::cout << std::flush;
                                    // }
                                }
                            }
                        }

                        // NOTE: Once a block that fits is found, because the domain's dimension are power of two, we
                        // know that we wont need an other block size. We can exit early.

                        KOLIOP_ASSERT(((local_domain_tor1_extent - block_count_in_tor1 * current_tor1_block_extent) == 0 &&
                                       (local_domain_tor2_extent - block_count_in_tor2 * current_tor2_block_extent) == 0 &&
                                       (local_domain_tor3_extent - block_count_in_tor3 * current_tor3_block_extent) == 0) &&
                                      "Expecting powers of two as space dimension.");
                        break;
                    }

                    // NOTE: The block wont fit, go to the next (smaller) block
                    // size.
                }

                KOLIOP_ASSERT(!cvcd_drivers_configurations.empty());
            }

            // NOTE: Sort of factory. I put in into an other translation unit
            // to mitigate the long compilation time (yay for c++ templates).
            void CreateCvCdDriver(const Operator&   an_operator,
                                  CvCdDriverEraser& a_cvcd_driver,
                                  OrdinalType       block_tor1_extent,
                                  OrdinalType       block_tor2_extent,
                                  OrdinalType       block_tor3_extent);

        } // namespace detail

        ////////////////////////////////////////////////////////////////////////
        // CvCdDriverScheduler methods definition
        ////////////////////////////////////////////////////////////////////////

        void CvCdDriverScheduler::Plan(const Operator& an_operator) {

            // NOTE: We assume preset_block_extent are sorted from larger to
            // smaller. This will not affect the correctness of the algorithm as
            // long as the minimum block size (of which all block are a
            // multiple) is present in the set. that said, this could affect the
            // performance of the cvcd driver. For instance if we use only small
            // block size where we could have used bigger ones.

            cvcd_drivers_configurations_.clear();

            detail::PrepareBlockPosition(an_operator.tor1_extent_,
                                         an_operator.tor2_extent_,
                                         an_operator.tor3_extent_,
                                         cvcd_drivers_configurations_);

            // NOTE:
            // - A larger block take more memory.
            // - By definition, we only need one block size to pave the plane.
            // - We have at most kDisjointExecutionSpaceCount disjoint execution
            // spaces.
            // - Drivers using different execution space MUST have disjoint
            // state!
            // - Knowing that, we can allocate only kDisjointExecutionSpaceCount
            // drivers (of the same size) and simply ventilate theses driver on
            // the execution spaces. We use shared pointer to share the drivers.
            // We can guarantee that at most something proportionate to
            // kDisjointExecutionSpaceCount * BlockSize will be allocated.
            // Shared instances of a driver share the same execution space. This
            // basically serialize the work from the POV of a driver preventing
            // race condition.

            const auto& execution_space_pool = an_operator.execution_space_pool_;

            KOLIOP_ASSERT(!execution_space_pool.empty());

            // The first (at most) execution_space_pool.size() get a freshly
            // allocated CvCdDriver and an execution space. The other block will
            // be copies of these ones.
            const OrdinalType original_block_count = std::min(cvcd_drivers_configurations_.size(),
                                                              execution_space_pool.size());

            for(OrdinalType i = 0; i < original_block_count; ++i) {
                auto& a_driver_configuration = cvcd_drivers_configurations_[i];

                detail::CreateCvCdDriver(an_operator,
                                         a_driver_configuration.cvcd_driver_,
                                         a_driver_configuration.block_tor1_extent_,
                                         a_driver_configuration.block_tor2_extent_,
                                         a_driver_configuration.block_tor3_extent_);

                // The modulo would be redundant thanks to
                // original_block_count.
                const OrdinalType execution_space_index = i /* % execution_space_pool.size() */;

                a_driver_configuration.execution_space_ = execution_space_pool[execution_space_index];
            }

            for(OrdinalType i = original_block_count;
                i < static_cast<OrdinalType>(cvcd_drivers_configurations_.size());
                ++i) {
                auto&       a_driver_configuration           = cvcd_drivers_configurations_[i];
                const auto& an_original_driver_configuration = cvcd_drivers_configurations_[i % original_block_count];

                // NOTE: We copy, but the CvCdDriverEraser uses shared pointer
                // which *prevents* deep copying.

                // NOTE: We copy the execution space and the driver but NOT
                // block_first_tor1_, block_first_tor2_ and
                // block_first_tor3_. These are and must stay different.
                a_driver_configuration.cvcd_driver_     = an_original_driver_configuration.cvcd_driver_;
                a_driver_configuration.execution_space_ = an_original_driver_configuration.execution_space_;
            }
        }

        void CvCdDriverScheduler::AllocateScratchMemory(const Operator& an_operator) {
            const auto&       execution_space_pool = an_operator.execution_space_pool_;
            const OrdinalType original_block_count = std::min(cvcd_drivers_configurations_.size(),
                                                              execution_space_pool.size());

            for(OrdinalType i = 0; i < original_block_count; ++i) {
                auto& a_driver_configuration = cvcd_drivers_configurations_[i];
                a_driver_configuration.cvcd_driver_->AllocateScratchMemory();
            }
        }

        void CvCdDriverScheduler::ReleaseScratchMemory(const Operator& an_operator) {
            const auto&       execution_space_pool = an_operator.execution_space_pool_;
            const OrdinalType original_block_count = std::min(cvcd_drivers_configurations_.size(),
                                                              execution_space_pool.size());
            for(OrdinalType i = 0; i < original_block_count; ++i) {
                auto& a_driver_configuration = cvcd_drivers_configurations_[i];
                a_driver_configuration.cvcd_driver_->ReleaseScratchMemory();
            }
        }

        void CvCdDriverScheduler::ScheduleExecution(const Operator&                           an_operator,
                                                    ComputeFloatType                          dt,
                                                    OrdinalType                               species_index,
                                                    const koliop::utility::View::UDL6<Flo64>& f5d_s) {
            for(auto&& a_driver_configuration : cvcd_drivers_configurations_) {
                a_driver_configuration.cvcd_driver_->Apply(an_operator,
                                                           a_driver_configuration.execution_space_,
                                                           a_driver_configuration.block_first_tor1_,
                                                           a_driver_configuration.block_first_tor2_,
                                                           a_driver_configuration.block_first_tor3_,
                                                           dt,
                                                           species_index,
                                                           f5d_s);
            }
        }

    } // namespace collisions
} // namespace koliop
