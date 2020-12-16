#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace picongpu
{
    namespace openPMD
    {
        template<typename Clock>
        struct TimeFormatters
        {
            using TimeFormatter = std::function<std::string(typename Clock::time_point const&)>;

            static std::string humanReadable(typename Clock::time_point const& currentTime)
            {
                std::stringstream res;
                std::time_t now_c = Clock::to_time_t(currentTime);
                res << std::put_time(std::localtime(&now_c), "%F %T") << '.' << std::setfill('0') << std::setw(3)
                    << std::chrono::duration_cast<std::chrono::milliseconds>(currentTime.time_since_epoch()).count()
                        % 1000;
                return res.str();
            }

            static std::string epochTime(typename Clock::time_point const& currentTime)
            {
                return std::to_string(
                    std::chrono::duration_cast<std::chrono::milliseconds>(currentTime.time_since_epoch()).count());
            }
        };
        template<typename Clock>
        using TimeFormatter = typename TimeFormatters<Clock>::TimeFormatter;

        // https://en.cppreference.com/w/cpp/named_req/Clock
        template<typename Clock = std::chrono::system_clock, bool enable = true>
        class DumpTimes
        {
        public:
            using time_point = typename Clock::time_point;
            using duration = typename Clock::duration;
            using Ret_T = std::pair<time_point, duration>;

            constexpr static char const* ENV_VAR = "PICONGPU_TIME_TRACE_FILE";
            const std::string filename;

            DumpTimes();
            DumpTimes(std::string filename);

            template<typename Duration>
            Ret_T now(
                std::string description,
                std::string separator = "\t",
                TimeFormatter<Clock> = &TimeFormatters<Clock>::epochTime);

            void append(std::string const&);

            void flush();

        private:
            std::fstream outStream;
            bool pendingNewline = false;
            time_point lastTimePoint;
        };

        template<typename Clock, bool enable>
        DumpTimes<Clock, enable>::DumpTimes() : DumpTimes(std::string(std::getenv(ENV_VAR)))
        {
        }

        template<typename Clock, bool enable>
        DumpTimes<Clock, enable>::DumpTimes(std::string _filename)
            : filename(std::move(_filename))
            , outStream(filename, std::ios_base::out | std::ios_base::app)
            , lastTimePoint(Clock::now())
        {
        }

        template<typename Clock, bool enable>
        template<typename Duration>
        typename DumpTimes<Clock, enable>::Ret_T DumpTimes<Clock, enable>::now(
            std::string description,
            std::string separator,
            TimeFormatter<Clock> timeFormatter)
        {
            auto currentTime = Clock::now();
            auto delta = currentTime - lastTimePoint;
            lastTimePoint = currentTime;
            std::string nowFormatted = timeFormatter(currentTime);
            if(pendingNewline)
            {
                outStream << '\n';
            }
            outStream << std::move(nowFormatted) << separator << std::chrono::duration_cast<Duration>(delta).count()
                      << separator << description;
            pendingNewline = true;
            return std::make_pair(currentTime, delta);
        }

        template<typename Clock, bool enable>
        void DumpTimes<Clock, enable>::append(std::string const& str)
        {
            outStream << str;
        }

        template<typename Clock, bool enable>
        void DumpTimes<Clock, enable>::flush()
        {
            if(pendingNewline)
            {
                outStream << '\n';
            }
            pendingNewline = false;
            outStream << std::flush;
        }

        template<typename Clock>
        class DumpTimes<Clock, false>
        {
        public:
            using Ret_T = void;
            DumpTimes()
            {
            }
            DumpTimes(std::string)
            {
            }

            template<typename, typename... Args>
            inline Ret_T now(Args&&...)
            {
            }

            template<typename... Args>
            inline void append(Args&&...)
            {
            }

            template<typename, typename... Args>
            inline void flush(Args&&...)
            {
            }
        };

    } // namespace openPMD
} // namespace picongpu
