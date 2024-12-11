// SPDX-License-Identifier: MIT

#ifndef KOLIOP_UTILITY_STOPWATCH_H
#define KOLIOP_UTILITY_STOPWATCH_H

#include <chrono>

namespace koliop {
    namespace utility {

        ////////////////////////////////////////////////////////////////////////
        // Stopwatch definition
        ////////////////////////////////////////////////////////////////////////

        /// Choose your duration, can be seconds or ms, or ns etc..
        /// https://en.cppreference.com/w/cpp/chrono/duration
        ///
        template <typename Duration = std::chrono::nanoseconds>
        class Stopwatch {
        protected:
            using time_point = std::chrono::steady_clock::time_point;

        public:
            /// Nanoseconds by default
            ///
            using duration = Duration;

            static_assert(std::chrono::steady_clock::is_steady, "Only steady clocks (the ones that only go forward) !");

        public:
            /// Start the Stopwatch
            ///
            Stopwatch();

            /// Return the Return time difference (duration) between the last Lap() or Reset() or object construction.
            /// Do not call Elapsed() and then Reset(), just call Lap() instead.
            ///
            duration Elapsed() const;

            /// Same as Elapsed() but return true if the Elapsed() time was >= duration, else false.
            /// Nifty idea by Marcelo Juchem.
            ///
            template <typename ADuration>
            bool Elapsed(ADuration&& duration) const;

            /// Same as lap but do not return the time difference (duration).
            /// Do not call Elapsed() and then Reset(), just call Lap() instead.
            ///
            void Reset();

            /// Return time difference (duration) between the last Lap() or Reset() or object construction.
            /// The next Lap() will be computed based on now. (that is, now - past_now).
            ///
            duration Lap();

            /// Same as Lap() but return true if the Elapsed() time was >= duration, else false.
            /// Nifty idea by Marcelo Juchem.
            ///
            template <typename ADuration>
            bool Lap(ADuration&& duration);

            template <typename Callable>
            static duration Measure(Callable&& a_callable);

            static duration ExperimentalResolution();

        protected:
            time_point last_point_in_time_;
        };

        ////////////////////////////////////////////////////////////////////////
        // Stopwatch methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Duration>
        Stopwatch<Duration>::Stopwatch()
            : last_point_in_time_{std::chrono::steady_clock::now()} {
            // EMPTY
        }

        template <typename Duration>
        typename Stopwatch<Duration>::duration
        Stopwatch<Duration>::Elapsed() const {
            // The cast is a no-op if std::chrono::steady_clock::duration "is same" duration. For other duration, I dunno.
            return std::chrono::duration_cast<duration>(std::chrono::steady_clock::now() - last_point_in_time_);
        }

        template <typename Duration>
        template <typename ADuration>
        bool Stopwatch<Duration>::Elapsed(ADuration&& duration) const {
            return (std::chrono::steady_clock::now() - last_point_in_time_) >= duration;
        }

        template <typename Duration>
        void Stopwatch<Duration>::Reset() {
            last_point_in_time_ = std::chrono::steady_clock::now();
        }

        template <typename Duration>
        typename Stopwatch<Duration>::duration
        Stopwatch<Duration>::Lap() {
            const time_point previous_point_in_time_ = last_point_in_time_;

            last_point_in_time_ = std::chrono::steady_clock::now();

            // The cast is a no-op if std::chrono::steady_clock::duration "is same" duration. For other duration, I dunno.
            return std::chrono::duration_cast<duration>(last_point_in_time_ - previous_point_in_time_);
        }

        template <typename Duration>
        template <typename ADuration>
        bool Stopwatch<Duration>::Lap(ADuration&& duration) {
            const time_point new_point_in_time = std::chrono::steady_clock::now();

            // Dont use Elapsed() because we want to be precise, some time may pass between the call to Elapsed() and
            // the next (necessary) call to now(). Think transactional.
            if(/* __builtin_expect((expr), 0) would that be useful ? */ (new_point_in_time - last_point_in_time_) >= duration) {
                last_point_in_time_ = new_point_in_time;
                return true;
            }

            return false;
        }

        template <typename Duration>
        template <typename Callable>
        typename Stopwatch<Duration>::duration
        Stopwatch<Duration>::Measure(Callable&& a_callable) {
            Stopwatch a_stopwatch;
            { a_callable(); }
            const auto duration = a_stopwatch.Elapsed();
            // std::cout << __PRETTY_FUNCTION__ << " took " << (duration.count() / 1'000) << "us\n";
            return duration;
        }

    } // namespace utility
} // namespace koliop

#endif
