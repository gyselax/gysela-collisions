// SPDX-License-Identifier: MIT
//
// Small sanity and bit-reproducibility test for koliop::collisions::Operator.
//
// The operator is fed physically meaningful inputs:
//   - two species (main ion A=1 Z=1, impurity A=4 Z=2) with radial n(r), T(r), U(r) profiles;
//   - a large aspect ratio field B(theta, r) = 1 / (1 + eps(r) cos(theta)) and Bstar = B (1 + rho * A/Z * vpar);
//   - a radial collisionality profile AD(r) ~ eps(r)^(3/2), zeroed in a 2 point radial buffer;
//   - trapezoid quadrature weights in mu and vpar;
//   - f a shifted Maxwellian in koliop's normalisation, FM = n / (2 pi T)^(3/2) exp(-(mu B + (vpar - U)^2 / 2) / T),
//     with a (theta, phi) density modulation and a small, bounded non-Maxwellian perturbation.
//
// For each configuration the test checks:
//   1. f stays finite and non-negative and the density is conserved within kDensityTolerance;
//   2. bit reproducibility: the same inputs give bitwise identical f and diagnostics (n, U_par, T, nu_ab, q_par/nT) for
//      (a) a second, independent operator and (b) the same operator reused on fresh input;
//   3. optionally, against stored references:
//        collisions_small_sanity --write-reference <file>   writes the digests to <file> and the raw arrays to
//                                                           <file>.<configuration>.<array>.bin,
//        collisions_small_sanity --reference <file>         compares the digests (bitwise) and, on a mismatch, reports
//                                                           error metrics against the raw arrays when available.
//      Digests are only expected to match for the same toolchain, backend and Kokkos configuration.
//      Add --skip-f5d to either option to not write / read the raw f5d_s file (by far the largest); its digest is still
//      written / compared, only the error metrics of a mismatch are then unavailable for f5d_s.
//
// Whenever two arrays are not bitwise identical, the error metrics of the (gyselalibxx collision) HDF5 validation tool
// are reported (L2 norms, relative L2 error, max absolute/relative errors and their location, non-finite values) and a
// warning is printed when they exceed the kRelativeErrorNorm2Tolerance, kMaxAbsoluteErrorTolerance and
// kMaxRelativeErrorTolerance tolerances. Any difference still fails the bit reproducibility checks; the metrics tell
// how large it is.
//
// Layout (tor1 = r, tor2 = theta, tor3 = phi): B_norm (theta, r, phi), Bstar_s (vpar, theta, r, phi, species), f5d_s
//   (mu, vpar, theta, r, phi, species), coeff_AD_r (r).
//
// NOTE: theta <= r on purpose. On devel d261ce8 nu.cc and cpar.cc index the r-sized nustar_coeff_ and coeff_AD_r with
// itor2 (theta), which reads out of bounds when theta > r. ExtractRaw(k_nustar_coeff) is not used for the same reason
// (wrong output extents when r != theta).

#include <KOLIOP/collisions.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace {

    using koliop::utility::View;
    using Diagnostics = koliop::collisions::Operator::Diagnostics;

    constexpr double kPi = 3.14159265358979323846;
    // Collisions() does not conserve the discrete density exactly: ComputeUparTCorrection (and, with interspecies
    // collisions, ComputeMaxwellianEvolution) add and remove analytically normalised Maxwellians whose quadrature
    // integrals differ. The residual is O(h^4) in the mu/vpar grid step relative to the local thermal scale; it is
    // ~3e-3 after 2 steps for the 32 x 64 grids below (and ~1e-4 at 64 x 128). The tolerance only catches gross errors.
    constexpr double kDensityTolerance = 1.0e-2;

    // Same tolerances as the HDF5 validation tool.
    constexpr double kRelativeErrorNorm2Tolerance = 1.0e-10;
    constexpr double kMaxAbsoluteErrorTolerance   = 1.0e-10;
    constexpr double kMaxRelativeErrorTolerance   = 1.0e-10;

    struct Configuration {
        std::size_t mu_extent;
        std::size_t vpar_extent;
        std::size_t r_extent;
        std::size_t theta_extent;
        std::size_t phi_extent;
        double      dt;
        int         step_count;

        /// Unique, self describing name. Used as the digest key and reference file suffix.
        std::string Name() const {
            char name[128];
            std::snprintf(name, sizeof(name), "mu%zu_vpar%zu_r%zu_theta%zu_phi%zu_dt%g_steps%d",
                          mu_extent, vpar_extent, r_extent, theta_extent, phi_extent, dt, step_count);
            return name;
        }
    };

    constexpr std::size_t kSpeciesExtent         = 2;
    constexpr double      kHatAs[kSpeciesExtent] = {1.0, 4.0};
    constexpr double      kHatZs[kSpeciesExtent] = {1.0, 2.0};
    constexpr double      kN0[kSpeciesExtent]    = {1.0, 0.05};
    constexpr double      kT0[kSpeciesExtent]    = {1.0, 1.2};
    constexpr double      kU0[kSpeciesExtent]    = {0.1, -0.05};

    /// (dimension name, extent) pairs, left most dimension is the fastest varying (layout left).
    using Extents = std::vector<std::pair<const char*, std::size_t>>;

    /// FNV-1a hash of the raw bytes: a compact bitwise fingerprint of an array.
    std::uint64_t Digest(const std::vector<double>& values) {
        // FNV-1a over the raw bytes: any bit flip changes the digest.
        std::uint64_t hash  = 0xcbf29ce484222325ULL;
        const auto*   bytes = reinterpret_cast<const unsigned char*>(values.data());
        for(std::size_t index = 0; index < values.size() * sizeof(double); ++index) {
            hash = (hash ^ bytes[index]) * 0x100000001b3ULL;
        }
        return hash;
    }

    /// Host -> device copy: allocates a device view and fills it through a host mirror (create_mirror_view + deep_copy).
    Kokkos::View<double*> ToDevice(const std::vector<double>& host, const char* label) {
        Kokkos::View<double*> device{label, host.size()};
        const auto            mirror = Kokkos::create_mirror_view(device);
        for(std::size_t index = 0; index < host.size(); ++index) {
            mirror(index) = host[index];
        }
        Kokkos::deep_copy(device, mirror);
        return device;
    }

    /// Device -> host copy (create_mirror_view_and_copy) into a plain std::vector.
    std::vector<double> ToHost(const Kokkos::View<double*>& device) {
        const auto          mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device);
        std::vector<double> host(mirror.extent(0));
        for(std::size_t index = 0; index < host.size(); ++index) {
            host[index] = mirror(index);
        }
        return host;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Error metrics, mirroring the HDF5 validation tool in gyselalibxx's collision validation
    ////////////////////////////////////////////////////////////////////////////

    /// Neumaier compensated summation: keeps the norms of large arrays accurate to a few ulps instead of
    /// accumulating O(size * epsilon) rounding error.
    struct CompensatedSum {
    public:
        void Add(double value) {
            const double total = sum_ + value;
            if(std::abs(sum_) >= std::abs(value)) {
                compensation_ += (sum_ - total) + value;
            } else {
                compensation_ += (value - total) + sum_;
            }
            sum_ = total;
        }

        double Value() const {
            return sum_ + compensation_;
        }

    protected:
        double sum_          = 0.0;
        double compensation_ = 0.0;
    };

    /// Output of ComputeErrorMetrics: the quantities printed by the validation tool.
    struct ErrorMetrics {
        bool        is_comparable = false; // False on size mismatch, empty arrays or no entry finite in both.
        double      reference_norm2;
        double      approximation_norm2;
        double      difference_norm2;
        double      relative_error_norm2;
        double      max_absolute_error;
        double      max_relative_error_at_max_absolute_error_location;
        std::size_t max_absolute_error_index;
        double      max_relative_error;
        double      max_absolute_error_at_max_relative_error_location;
        std::size_t max_relative_error_index;
        bool        has_degenerated_reference     = false;
        bool        has_degenerated_approximation = false;
        bool        warning                       = true;
    };

    /// Only entries finite in both arrays enter the metrics. A zero reference with a non zero error gives an infinite
    /// relative error. Ties in the max searches resolve to the first index, like numpy.argmax.
    ErrorMetrics ComputeErrorMetrics(const std::vector<double>& reference, const std::vector<double>& approximation) {
        ErrorMetrics metrics;
        if(reference.size() != approximation.size() || reference.empty()) {
            return metrics;
        }

        CompensatedSum reference_sum2;
        CompensatedSum approximation_sum2;
        CompensatedSum difference_sum2;
        bool           has_valid_entry = false;

        for(std::size_t index = 0; index < reference.size(); ++index) {
            const bool is_reference_finite     = std::isfinite(reference[index]);
            const bool is_approximation_finite = std::isfinite(approximation[index]);

            metrics.has_degenerated_reference |= !is_reference_finite;
            metrics.has_degenerated_approximation |= !is_approximation_finite;

            if(!is_reference_finite || !is_approximation_finite) {
                continue;
            }

            const double difference         = approximation[index] - reference[index];
            const double absolute_error     = std::abs(difference);
            const double absolute_reference = std::abs(reference[index]);

            double relative_error = 0.0;
            if(absolute_reference != 0.0) {
                relative_error = absolute_error / absolute_reference;
            } else if(absolute_error != 0.0) {
                relative_error = std::numeric_limits<double>::infinity();
            }

            reference_sum2.Add(reference[index] * reference[index]);
            approximation_sum2.Add(approximation[index] * approximation[index]);
            difference_sum2.Add(difference * difference);

            if(!has_valid_entry || absolute_error > metrics.max_absolute_error) {
                metrics.max_absolute_error                                = absolute_error;
                metrics.max_relative_error_at_max_absolute_error_location = relative_error;
                metrics.max_absolute_error_index                          = index;
            }
            if(!has_valid_entry || relative_error > metrics.max_relative_error) {
                metrics.max_relative_error                                = relative_error;
                metrics.max_absolute_error_at_max_relative_error_location = absolute_error;
                metrics.max_relative_error_index                          = index;
            }
            has_valid_entry = true;
        }

        if(!has_valid_entry) {
            return metrics;
        }

        metrics.is_comparable       = true;
        metrics.reference_norm2     = std::sqrt(reference_sum2.Value());
        metrics.approximation_norm2 = std::sqrt(approximation_sum2.Value());
        metrics.difference_norm2    = std::sqrt(difference_sum2.Value());

        if(metrics.reference_norm2 == 0.0) {
            metrics.relative_error_norm2 = metrics.difference_norm2 == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
        } else {
            metrics.relative_error_norm2 = metrics.difference_norm2 / metrics.reference_norm2;
        }

        metrics.warning = metrics.has_degenerated_reference ||
                          metrics.has_degenerated_approximation ||
                          metrics.relative_error_norm2 > kRelativeErrorNorm2Tolerance ||
                          (metrics.max_absolute_error > kMaxAbsoluteErrorTolerance &&
                           metrics.max_relative_error > kMaxRelativeErrorTolerance);
        return metrics;
    }

    /// Formats a flat (layout left) index as "(mu=3, vpar=10, ...)".
    std::string FormatIndex(std::size_t flat_index, const Extents& extents) {
        std::string formatted = "(";
        for(std::size_t dimension = 0; dimension < extents.size(); ++dimension) {
            const auto& [dimension_name, extent] = extents[dimension];
            formatted += std::string{dimension == 0 ? "" : ", "} + dimension_name + "=" + std::to_string(flat_index % extent);
            flat_index /= extent;
        }
        return formatted + ")";
    }

    /// Prints the metrics in the validation tool format; name labels the array.
    void PrintErrorMetrics(const std::string& name, const Extents& extents, const ErrorMetrics& metrics) {
        if(!metrics.is_comparable) {
            std::printf("    [WARNING]: '%s' cannot be compared (size mismatch, empty, or no entry finite in both)\n", name.c_str());
            return;
        }
        if(metrics.warning) {
            std::printf("    [WARNING]: Non conformity past tolerances detected!\n");
        }
        std::printf("    NORM2 (r, a, a-r)           : %s %.17e %.17e %.17e\n", name.c_str(),
                    metrics.reference_norm2, metrics.approximation_norm2, metrics.difference_norm2);
        std::printf("    NORM2(a-r) / NORM2(r)       : %s %.17e\n", name.c_str(), metrics.relative_error_norm2);
        std::printf("    MAXVAL(ABS(a-r))            : %s %.17e ABS(a-r)/ABS(r) at argmax ABS(a-r): %.17e\n", name.c_str(),
                    metrics.max_absolute_error, metrics.max_relative_error_at_max_absolute_error_location);
        std::printf("    MAXVAL(ABS(a-r)/ABS(r))     : %s %.17e ABS(a-r) at argmax relative error: %.17e\n", name.c_str(),
                    metrics.max_relative_error, metrics.max_absolute_error_at_max_relative_error_location);
        std::printf("    Degenerated reference       : %s %s\n", name.c_str(), metrics.has_degenerated_reference ? "True" : "False");
        std::printf("    Degenerated approximation   : %s %s\n", name.c_str(), metrics.has_degenerated_approximation ? "True" : "False");
        std::printf("    Max absolute error index    : %s\n", FormatIndex(metrics.max_absolute_error_index, extents).c_str());
        std::printf("    Max relative error index    : %s\n", FormatIndex(metrics.max_relative_error_index, extents).c_str());
    }

    ////////////////////////////////////////////////////////////////////////////
    // Operator inputs and runs
    ////////////////////////////////////////////////////////////////////////////

    /// The host side operator inputs and initial distribution function: flat std::vector in layout left order, no device
    /// copy yet (see OperatorInstance).
    struct Inputs {
        std::vector<double> hat_As, hat_Zs, mug, vparg, coeff_intdmu, coeff_intdvpar, coeff_AD_r, B_norm, Bstar_s, f5d_s;

        /// Builds grids, quadrature weights, profiles, B / Bstar and the perturbed shifted Maxwellian on the host.
        explicit Inputs(const Configuration& configuration) {
            const std::size_t mu = configuration.mu_extent, vp = configuration.vpar_extent;
            const std::size_t nr = configuration.r_extent, nth = configuration.theta_extent, nph = configuration.phi_extent;
            const std::size_t ns = kSpeciesExtent;

            hat_As.assign(kHatAs, kHatAs + ns);
            hat_Zs.assign(kHatZs, kHatZs + ns);

            const double mu_max = 10.0, vpar_max = 5.0;
            mug.resize(mu);
            coeff_intdmu.resize(mu);
            for(std::size_t imu = 0; imu < mu; ++imu) {
                mug[imu]          = mu_max * static_cast<double>(imu) / static_cast<double>(mu - 1);
                coeff_intdmu[imu] = (mu_max / static_cast<double>(mu - 1)) * ((imu == 0 || imu == mu - 1) ? 0.5 : 1.0);
            }
            vparg.resize(vp);
            coeff_intdvpar.resize(vp);
            for(std::size_t iv = 0; iv < vp; ++iv) {
                vparg[iv]          = -vpar_max + 2.0 * vpar_max * static_cast<double>(iv) / static_cast<double>(vp - 1);
                coeff_intdvpar[iv] = (2.0 * vpar_max / static_cast<double>(vp - 1)) * ((iv == 0 || iv == vp - 1) ? 0.5 : 1.0);
            }

            const auto normalized_radius = [&](std::size_t ir) {
                return (static_cast<double>(ir) + 0.5) / static_cast<double>(nr);
            };
            const auto inverse_aspect_ratio = [&](std::size_t ir) {
                return 0.05 + 0.15 * normalized_radius(ir);
            };
            const auto magnetic_field = [&](std::size_t ith, std::size_t ir) {
                return 1.0 / (1.0 + inverse_aspect_ratio(ir) * std::cos(2.0 * kPi * static_cast<double>(ith) / static_cast<double>(nth)));
            };

            coeff_AD_r.resize(nr);
            for(std::size_t ir = 0; ir < nr; ++ir) {
                // Collisions are switched off in the outer 2 point buffer.
                coeff_AD_r[ir] = (ir + 2 < nr) ? 1.0e-2 * std::pow(inverse_aspect_ratio(ir), 1.5) : 0.0;
            }

            const auto space_index = [&](std::size_t ith, std::size_t ir, std::size_t iph) {
                return ith + nth * (ir + nr * iph);
            };
            const std::size_t space = nth * nr * nph;

            B_norm.resize(space);
            Bstar_s.resize(vp * space * ns);
            for(std::size_t iph = 0; iph < nph; ++iph) {
                for(std::size_t ir = 0; ir < nr; ++ir) {
                    for(std::size_t ith = 0; ith < nth; ++ith) {
                        B_norm[space_index(ith, ir, iph)] = magnetic_field(ith, ir);
                        for(std::size_t ispecies = 0; ispecies < ns; ++ispecies) {
                            for(std::size_t iv = 0; iv < vp; ++iv) {
                                Bstar_s[iv + vp * (space_index(ith, ir, iph) + space * ispecies)] =
                                    magnetic_field(ith, ir) * (1.0 + 0.01 * (hat_As[ispecies] / hat_Zs[ispecies]) * vparg[iv]);
                            }
                        }
                    }
                }
            }

            f5d_s.resize(mu * vp * space * ns);
            for(std::size_t ispecies = 0; ispecies < ns; ++ispecies) {
                for(std::size_t iph = 0; iph < nph; ++iph) {
                    for(std::size_t ir = 0; ir < nr; ++ir) {
                        for(std::size_t ith = 0; ith < nth; ++ith) {
                            const double theta             = 2.0 * kPi * static_cast<double>(ith) / static_cast<double>(nth);
                            const double phi               = 2.0 * kPi * static_cast<double>(iph) / static_cast<double>(nph);
                            const double density           = kN0[ispecies] * (1.0 - 0.4 * normalized_radius(ir) * normalized_radius(ir)) * (1.0 + 0.02 * std::cos(theta + phi));
                            const double temperature       = kT0[ispecies] * (1.0 - 0.2 * normalized_radius(ir));
                            const double parallel_velocity = kU0[ispecies] * (1.0 - normalized_radius(ir));
                            const double field             = magnetic_field(ith, ir);
                            for(std::size_t iv = 0; iv < vp; ++iv) {
                                for(std::size_t imu = 0; imu < mu; ++imu) {
                                    const double shifted_vpar = vparg[iv] - parallel_velocity;
                                    const double maxwellian   = density * std::pow(2.0 * kPi * temperature, -1.5) *
                                                              std::exp(-(mug[imu] * field + 0.5 * shifted_vpar * shifted_vpar) / temperature);
                                    // Bounded (|.| <= 0.05) non-Maxwellian perturbation so that the operator has work
                                    // to do.
                                    const double perturbation = 0.05 * std::sin(shifted_vpar / std::sqrt(temperature)) *
                                                                std::exp(-0.5 * mug[imu] * field / temperature);
                                    f5d_s[imu + mu * (iv + vp * (space_index(ith, ir, iph) + space * ispecies))] = maxwellian * (1.0 + perturbation);
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    /// Device copies of the inputs plus the operator built on them.
    /// Host -> device copy happens once per input (ToDevice). The operator is then given unmanaged views (pointer +
    /// extents) on these arrays, so no data is copied there, and the instance must outlive the operator calls.
    struct OperatorInstance {
        Kokkos::View<double*>        hat_As, hat_Zs, mug, vparg, coeff_intdmu, coeff_intdvpar, coeff_AD_r, B_norm, Bstar_s, f5d_s;
        koliop::collisions::Operator an_operator;

        OperatorInstance(const Configuration& configuration, const Inputs& inputs)
            : hat_As{ToDevice(inputs.hat_As, "hat_As")}
            , hat_Zs{ToDevice(inputs.hat_Zs, "hat_Zs")}
            , mug{ToDevice(inputs.mug, "mug")}
            , vparg{ToDevice(inputs.vparg, "vparg")}
            , coeff_intdmu{ToDevice(inputs.coeff_intdmu, "coeff_intdmu")}
            , coeff_intdvpar{ToDevice(inputs.coeff_intdvpar, "coeff_intdvpar")}
            , coeff_AD_r{ToDevice(inputs.coeff_AD_r, "coeff_AD_r")}
            , B_norm{ToDevice(inputs.B_norm, "B_norm")}
            , Bstar_s{ToDevice(inputs.Bstar_s, "Bstar_s")}
            , f5d_s{ToDevice(inputs.f5d_s, "f5d_s")}
            , an_operator{configuration.r_extent, configuration.theta_extent, configuration.phi_extent, /* collision_interspecies */ true,
                          View::UDL1<const double>{hat_As.data(), kSpeciesExtent},
                          View::UDL1<const double>{hat_Zs.data(), kSpeciesExtent},
                          View::UDL1<const double>{mug.data(), configuration.mu_extent},
                          View::UDL1<const double>{vparg.data(), configuration.vpar_extent},
                          View::UDL1<const double>{coeff_intdmu.data(), configuration.mu_extent},
                          View::UDL1<const double>{coeff_intdvpar.data(), configuration.vpar_extent},
                          View::UDL1<const double>{coeff_AD_r.data(), configuration.r_extent},
                          View::UDL3<const double>{B_norm.data(), configuration.theta_extent, configuration.r_extent, configuration.phi_extent},
                          View::UDL5<const double>{Bstar_s.data(), configuration.vpar_extent, configuration.theta_extent, configuration.r_extent, configuration.phi_extent, kSpeciesExtent}} {}

        /// Unmanaged (mu, vpar, theta, r, phi, species) view on f5d_s, no copy.
        View::UDL6<double> F(const Configuration& configuration) {
            return View::UDL6<double>{f5d_s.data(), configuration.mu_extent, configuration.vpar_extent, configuration.theta_extent,
                                      configuration.r_extent, configuration.phi_extent, kSpeciesExtent};
        }

        /// ExtractRaw writes the diagnostic into a device buffer, ToHost then copies it to the host.
        std::vector<double> Extract(Diagnostics a_diagnostic, std::size_t size) {
            Kokkos::View<double*> buffer{"diagnostic", size};
            an_operator.ExtractRaw(a_diagnostic, buffer.data());
            an_operator.FenceAll();
            return ToHost(buffer);
        }
    };

    /// Everything that must be bit reproducible, by name.
    using Outputs = std::map<std::string, std::vector<double>>;

    /// The extents of each array in Outputs, used to locate errors.
    std::map<std::string, Extents> OutputExtents(const Configuration& configuration) {
        const Extents space{{"theta", configuration.theta_extent}, {"r", configuration.r_extent}, {"phi", configuration.phi_extent}};

        Extents moment = space;
        moment.emplace_back("species", kSpeciesExtent);

        Extents frequency = space;
        frequency.emplace_back("species_a", kSpeciesExtent);
        frequency.emplace_back("species_b", kSpeciesExtent);

        Extents distribution{{"mu", configuration.mu_extent}, {"vpar", configuration.vpar_extent}};
        distribution.insert(distribution.end(), moment.begin(), moment.end());

        return {{"f5d_s", distribution}, {"n_s", moment}, {"U_par_s", moment}, {"T_s", moment}, {"nu_ab", frequency}, {"q_par_on_nT_s", moment}};
    }

    /// Runs the collision steps on fresh input then gathers f and the diagnostics.
    /// Copies: fresh f host -> device (deep_copy into the instance f5d_s), then final f and each diagnostic
    /// device -> host (ToHost). density_drift receives max |n_after - n_before| / n_before.
    Outputs Run(const Configuration& configuration, const Inputs& inputs, OperatorInstance& instance, double* density_drift) {
        const std::size_t space = configuration.theta_extent * configuration.r_extent * configuration.phi_extent;

        Kokkos::deep_copy(instance.f5d_s, ToDevice(inputs.f5d_s, "f5d_s_fresh"));
        auto distribution = instance.F(configuration);

        instance.an_operator.ComputeMoments(distribution);
        const auto n_before = instance.Extract(Diagnostics::k_n_s, space * kSpeciesExtent);

        for(int step = 0; step < configuration.step_count; ++step) {
            instance.an_operator.Collisions(configuration.dt, distribution);
        }
        instance.an_operator.FenceAll();

        Outputs out;
        out["f5d_s"] = ToHost(instance.f5d_s);

        // Recompute the diagnostics on the final f. ComputeMoments may clamp f (to >= 1e-40) which is why f was
        // captured first.
        instance.an_operator.ComputeMoments(distribution);
        instance.an_operator.PrecomputeNu();
        instance.an_operator.ComputeQparOnNT(distribution);
        out["n_s"]           = instance.Extract(Diagnostics::k_n_s, space * kSpeciesExtent);
        out["U_par_s"]       = instance.Extract(Diagnostics::k_U_par_s, space * kSpeciesExtent);
        out["T_s"]           = instance.Extract(Diagnostics::k_T_s, space * kSpeciesExtent);
        out["nu_ab"]         = instance.Extract(Diagnostics::k_nu_ab, space * kSpeciesExtent * kSpeciesExtent);
        out["q_par_on_nT_s"] = instance.Extract(Diagnostics::k_q_par_on_nT_s, space * kSpeciesExtent);

        double drift = 0.0;
        for(std::size_t index = 0; index < n_before.size(); ++index) {
            drift = std::max(drift, std::abs(out["n_s"][index] - n_before[index]) / n_before[index]);
        }
        *density_drift = drift;
        return out;
    }

    /// Returns the number of arrays that are not bitwise identical. The error metrics are printed for each of them.
    int CompareBitwise(const char* label, const Outputs& reference, const Outputs& other, const std::map<std::string, Extents>& extents) {
        int mismatches = 0;
        for(const auto& [name, expected] : reference) {
            const auto& actual = other.at(name);
            if(expected.size() == actual.size() &&
               std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(double)) == 0) {
                continue;
            }
            ++mismatches;
            std::size_t differing_count = 0;
            for(std::size_t index = 0; index < std::min(expected.size(), actual.size()); ++index) {
                if(std::memcmp(&expected[index], &actual[index], sizeof(double)) != 0) {
                    ++differing_count;
                }
            }
            std::printf("    FAIL %s: %s differs in %zu / %zu values\n", label, name.c_str(), differing_count, expected.size());
            PrintErrorMetrics(name, extents.at(name), ComputeErrorMetrics(expected, actual));
        }
        return mismatches;
    }

    /// Raw array file attached to a reference digest file: <reference>.<configuration>.<array>.bin.
    std::string ReferenceDataPath(const std::string& reference_path, const std::string& key) {
        return reference_path + "." + key + ".bin";
    }

    /// Writes the raw doubles of a host array.
    bool WriteArray(const std::string& path, const std::vector<double>& values) {
        std::ofstream stream{path, std::ios::binary};
        stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
        return static_cast<bool>(stream);
    }

    /// Returns an empty vector if the file does not exist or is not made of doubles.
    std::vector<double> ReadArray(const std::string& path) {
        std::ifstream stream{path, std::ios::binary | std::ios::ate};
        if(!stream) {
            return {};
        }
        const auto byte_count = static_cast<std::size_t>(stream.tellg());
        if(byte_count % sizeof(double) != 0) {
            return {};
        }
        std::vector<double> values(byte_count / sizeof(double));
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(byte_count));
        return values;
    }

} // namespace

/// Per configuration: build inputs, run on two independent operators and once more on a reused one, check sanity and
/// bitwise equality, record digests; then optionally write or compare the reference files (see the header).
int main(int argc, char** argv) {
    std::string reference_path, write_reference_path;
    bool        skip_f5d = false;
    for(int iargument = 1; iargument < argc; ++iargument) {
        if(std::strcmp(argv[iargument], "--reference") == 0 && iargument + 1 < argc) {
            reference_path = argv[++iargument];
        } else if(std::strcmp(argv[iargument], "--write-reference") == 0 && iargument + 1 < argc) {
            write_reference_path = argv[++iargument];
        } else if(std::strcmp(argv[iargument], "--skip-f5d") == 0) {
            skip_f5d = true;
        }
    }
    const auto is_skipped = [&](const std::string& key) {
        return skip_f5d && key.size() >= 6 && key.compare(key.size() - 6, 6, ".f5d_s") == 0;
    };

    Initialize(8, 0, true);

    // NOTE:
    // - dt = 0 and dt = 0.01 give the same density drift (2.4e-4). That confirms the drift comes from the Laguerre
    //   projection and reconstruction, not from the collision terms.

    // {mu, vpar, r, theta, phi, dt, steps}
    const Configuration configurations[] = {
        {64, 128, 1, 1, 1, 1.0E-2, 1},
        {48, 64, 16, 8, 4, 1.0E-2, 1},
        {48, 64, 16, 8, 4, 1.0E-2, 2},
        {48, 64, 16, 8, 4, 0.0, 1},
        {48, 64, 16, 8, 4, 0.0, 2},
        {32, 64, 64, 32, 2, 1.0E-2, 2},
    };

    int failures = 0;

    std::map<std::string, std::uint64_t>       digests;
    std::map<std::string, std::vector<double>> arrays;
    std::map<std::string, Extents>             array_extents;

    for(const auto& configuration : configurations) {
        const std::string configuration_name = configuration.Name();
        std::printf("[%s] mu x vpar = %zu x %zu, r x theta x phi = %zu x %zu x %zu, species = %zu, %d step(s), dt = %g\n",
                    configuration_name.c_str(), configuration.mu_extent, configuration.vpar_extent, configuration.r_extent,
                    configuration.theta_extent, configuration.phi_extent, kSpeciesExtent, configuration.step_count, configuration.dt);
        std::fflush(stdout);

        const auto   extents = OutputExtents(configuration);
        const Inputs inputs{configuration};
        Outputs      first, second, reused;
        double       drift_first = 0.0, drift_second = 0.0, drift_reused = 0.0;
        {
            OperatorInstance instance{configuration, inputs};
            first = Run(configuration, inputs, instance, &drift_first);
        }
        {
            OperatorInstance instance{configuration, inputs};
            second = Run(configuration, inputs, instance, &drift_second);
            // Same operator, fresh f: catches state leaking between calls (e.g. the U_par_s buffer swap).
            reused = Run(configuration, inputs, instance, &drift_reused);
        }

        // 1. Sanity.
        std::size_t non_finite = 0, negative = 0;
        for(const double value : first["f5d_s"]) {
            non_finite += !std::isfinite(value);
            negative += value < 0.0;
        }
        const bool sane = non_finite == 0 && negative == 0 && drift_first < kDensityTolerance;
        std::printf("    %s sanity: non-finite = %zu, negative = %zu, max |dn|/n = %.3e (tolerance %.1e)\n",
                    sane ? "ok  " : "FAIL", non_finite, negative, drift_first, kDensityTolerance);
        failures += !sane;

        // 2. Bit reproducibility.
        const int independent = CompareBitwise("independent operator", first, second, extents);
        const int reuse       = CompareBitwise("reused operator", first, reused, extents);
        std::printf("    %s bitwise reproducible: independent operator %s, reused operator %s\n",
                    (independent + reuse) == 0 ? "ok  " : "FAIL", independent == 0 ? "yes" : "NO", reuse == 0 ? "yes" : "NO");
        failures += (independent + reuse) != 0;

        for(auto& [name, values] : first) {
            const std::string key = configuration_name + "." + name;
            digests[key]          = Digest(values);
            array_extents[key]    = extents.at(name);
            std::printf("    digest %-58s %016llx\n", key.c_str(), static_cast<unsigned long long>(digests[key]));
            if((!write_reference_path.empty() || !reference_path.empty()) && !is_skipped(key)) {
                arrays[key] = std::move(values);
            }
        }
        std::fflush(stdout);
    }

    // 3. Optional comparison against stored references.
    if(!write_reference_path.empty()) {
        std::ofstream out{write_reference_path};
        for(const auto& [key, digest] : digests) {
            char hex[17];
            std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(digest));
            out << key << " " << hex << "\n";
            if(!is_skipped(key) && !WriteArray(ReferenceDataPath(write_reference_path, key), arrays.at(key))) {
                std::printf("FAIL cannot write %s\n", ReferenceDataPath(write_reference_path, key).c_str());
                ++failures;
            }
        }
        std::printf("Reference digests written to %s (raw arrays to %s.<configuration>.<array>.bin)\n",
                    write_reference_path.c_str(), write_reference_path.c_str());
    }
    if(!reference_path.empty()) {
        std::ifstream reference_stream{reference_path};
        if(!reference_stream) {
            std::printf("FAIL cannot read reference %s\n", reference_path.c_str());
            ++failures;
        }
        std::string line;
        int         mismatches = 0, compared = 0;
        while(std::getline(reference_stream, line)) {
            std::istringstream fields{line};
            std::string        key, hex;
            if(!(fields >> key >> hex)) {
                continue;
            }
            const auto digest_iterator = digests.find(key);
            ++compared;
            if(digest_iterator != digests.end() && digest_iterator->second == std::stoull(hex, nullptr, 16)) {
                continue;
            }
            ++mismatches;
            if(digest_iterator == digests.end()) {
                std::printf("    FAIL reference mismatch for %s: not produced by this run\n", key.c_str());
                continue;
            }
            std::printf("    FAIL reference mismatch for %s\n", key.c_str());
            const auto reference_values = is_skipped(key) ? std::vector<double>{} : ReadArray(ReferenceDataPath(reference_path, key));
            if(reference_values.empty()) {
                std::printf("    (no raw reference data at %s, metrics unavailable)\n", ReferenceDataPath(reference_path, key).c_str());
            } else {
                PrintErrorMetrics(key, array_extents.at(key), ComputeErrorMetrics(reference_values, arrays.at(key)));
            }
        }
        std::printf("%s reference %s: %d / %d digest(s) match\n", mismatches == 0 ? "ok  " : "FAIL",
                    reference_path.c_str(), compared - mismatches, compared);
        failures += mismatches != 0;
    }

    Deinitialize();
    std::printf("%s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
