
        static void DoMultiply() {
            Kokkos::DefaultExecutionSpace a_queue{Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace{}, 1)[0]};

            Kokkos::View<double*,
                         Kokkos::DefaultExecutionSpace::memory_space>
                a_result{Kokkos::view_alloc(a_queue, Kokkos::WithoutInitializing, "b"),
                         16};

            auto policy_with_hint = Kokkos::Experimental::require(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(a_queue,
                                                                                                                     0,
                                                                                                                     1024 * 1024 * 1024),
                                                                  Kokkos::Experimental::WorkItemProperty::None);

            const auto Compute = KOKKOS_LAMBDA(int n) {
                const auto extent = 1024 * 64;

                // using Float          = float2;
                // Float       value_0  = Float{a_result((n + 0) % extent), a_result((n + 1) % extent)};
                // Float       value_1  = Float{a_result((n + 2) % extent), a_result((n + 3) % extent)};
                // Float       value_2  = Float{a_result((n + 4) % extent), a_result((n + 5) % extent)};
                // Float       value_3  = Float{a_result((n + 6) % extent), a_result((n + 7) % extent)};
                // Float       value_4  = Float{a_result((n + 8) % extent), a_result((n + 9) % extent)};
                // Float       value_5  = Float{a_result((n + 10) % extent), a_result((n + 11) % extent)};
                // Float       value_6  = Float{a_result((n + 12) % extent), a_result((n + 13) % extent)};
                // Float       value_7  = Float{a_result((n + 14) % extent), a_result((n + 15) % extent)};
                // const Float constant = Float{1.0F, 1.0F} * a_result((n + 16) % extent);

                using Float          = float; // or double
                Float       value_0  = Float(a_result((n + 0) % extent));
                Float       value_1  = Float(a_result((n + 1) % extent));
                Float       value_2  = Float(a_result((n + 2) % extent));
                Float       value_3  = Float(a_result((n + 3) % extent));
                Float       value_4  = Float(a_result((n + 4) % extent));
                Float       value_5  = Float(a_result((n + 5) % extent));
                Float       value_6  = Float(a_result((n + 6) % extent));
                Float       value_7  = Float(a_result((n + 7) % extent));
                const Float constant = a_result((n + 8) % extent);

                // double: 512
                // float:  512
                // float2: 1024
                for(std::size_t i = 0; i < 512; ++i) {
                    value_0 += value_0 * constant;
                    value_1 += value_1 * constant;
                    value_2 += value_2 * constant;
                    value_3 += value_3 * constant;
                    value_4 += value_4 * constant;
                    value_5 += value_5 * constant;
                    value_6 += value_6 * constant;
                    value_7 += value_7 * constant;
                }

                value_0 = ((value_0 + value_1) + (value_2 + value_3)) +
                          ((value_4 + value_5) + (value_6 + value_7));

                // a_result((n + 0) % extent) = value_0.data[0];
                // a_result((n + 1) % extent) = value_0.data[1];

                a_result((n + 0) % extent) = value_0;
            };

            Kokkos::parallel_for("okok",
                                 policy_with_hint,
                                 Compute);
            Kokkos::fence();
        }
