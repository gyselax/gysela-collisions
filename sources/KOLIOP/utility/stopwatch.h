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

            /// Same as Elapsed() but return true if the Elapsed() time was >= the_duration, else false.
            /// Nifty idea by Marcelo Juchem.
            ///
            template <typename ADuration>
            bool Elapsed(ADuration&& the_duration) const;

            /// Same as lap but do not return the time difference (duration).
            /// Do not call Elapsed() and then Reset(), just call Lap() instead.
            ///
            void Reset();

            /// Return time difference (duration) between the last Lap() or Reset() or object construction.
            /// The next Lap() will be computed based on now. (that is, now - the_past_now).
            ///
            duration Lap();

            /// Same as Lap() but return true if the Elapsed() time was >= the_duration, else false.
            /// Nifty idea by Marcelo Juchem.
            ///
            template <typename ADuration>
            bool Lap(ADuration&& the_duration);

            template <typename Callable>
            static duration Measure(Callable&& a_callable);

            static duration ExperimentalResolution();

        protected:
            time_point the_last_point_in_time_;
        };

        ////////////////////////////////////////////////////////////////////////
        // Stopwatch methods definition
        ////////////////////////////////////////////////////////////////////////

        template <typename Duration>
        Stopwatch<Duration>::Stopwatch()
            : the_last_point_in_time_{std::chrono::steady_clock::now()} {
            // EMPTY
        }

        template <typename Duration>
        typename Stopwatch<Duration>::duration
        Stopwatch<Duration>::Elapsed() const {
            // The cast is a noop if std::chrono::steady_clock::duration "is same" duration. For other duration, I dunno.
            return std::chrono::duration_cast<duration>(std::chrono::steady_clock::now() - the_last_point_in_time_);
        }

        template <typename Duration>
        template <typename ADuration>
        bool Stopwatch<Duration>::Elapsed(ADuration&& the_duration) const {
            return (std::chrono::steady_clock::now() - the_last_point_in_time_) >= the_duration;
        }

        template <typename Duration>
        void Stopwatch<Duration>::Reset() {
            the_last_point_in_time_ = std::chrono::steady_clock::now();
        }

        template <typename Duration>
        typename Stopwatch<Duration>::duration
        Stopwatch<Duration>::Lap() {
            const time_point the_previous_point_in_time_ = the_last_point_in_time_;

            the_last_point_in_time_ = std::chrono::steady_clock::now();

            // The cast is a noop if std::chrono::steady_clock::duration "is same" duration. For other duration, I dunno.
            return std::chrono::duration_cast<duration>(the_last_point_in_time_ - the_previous_point_in_time_);
        }

        template <typename Duration>
        template <typename ADuration>
        bool Stopwatch<Duration>::Lap(ADuration&& the_duration) {
            const time_point the_new_point_in_time = std::chrono::steady_clock::now();

            // Dont use Elapsed() because we want to be precise, some time may pass between the call to Elapsed() and
            // the next (necessary) call to now(). Think transactional.
            if(/* __builtin_expect((the_expr), 0) would that be useful ? */ (the_new_point_in_time - the_last_point_in_time_) >= the_duration) {
                the_last_point_in_time_ = the_new_point_in_time;
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
            const auto the_duration = a_stopwatch.Elapsed();
            // std::cout << __PRETTY_FUNCTION__ << " took " << (the_duration.count() / 1'000) << "us\n";
            return the_duration;
        }

    } // namespace utility
} // namespace koliop

#endif
