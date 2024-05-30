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

        // https://en.cppreference.com/w/cpp/named_req/Clock
        template<bool enable = true, typename Clock = std::chrono::system_clock>
        class DumpTimes
        {
        public:
            using time_point = typename Clock::time_point;
            using duration = typename Clock::duration;

            using TimeFormatter = typename TimeFormatters<Clock>::TimeFormatter;
            using Ret_t = std::pair<time_point, duration>;

            const std::string filename;

            DumpTimes(std::string filename);

            template<typename Duration>
            auto now(
                std::string description,
                std::string separator = "\t",
                TimeFormatter = &TimeFormatters<Clock>::epochTime) -> Ret_t;

            auto append(std::string const&) -> void;

            auto flush() -> void;

        private:
            std::fstream outStream;
            bool pendingNewline = false;
            time_point lastTimePoint;
        };

        template<bool enable, typename Clock>
        DumpTimes<enable, Clock>::DumpTimes(std::string _filename)
            : filename(std::move(_filename))
            , outStream(filename, std::ios_base::out | std::ios_base::app)
            , lastTimePoint(Clock::now())
        {
            outStream << "timestamp\tdifftime\tdescription";
            pendingNewline = true;
        }

        template<bool enable, typename Clock>
        template<typename Duration>
        auto DumpTimes<enable, Clock>::now(
            std::string description,
            std::string separator,
            TimeFormatter timeFormatter) -> Ret_t
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

        template<bool enable, typename Clock>
        auto DumpTimes<enable, Clock>::append(std::string const& str) -> void
        {
            outStream << str;
        }

        template<bool enable, typename Clock>
        auto DumpTimes<enable, Clock>::flush() -> void
        {
            if(pendingNewline)
            {
                outStream << '\n';
            }
            pendingNewline = false;
            outStream << std::flush;
        }

        template<typename Clock>
        class DumpTimes<false, Clock>
        {
        public:
            using Ret_t = void;
            DumpTimes()
            {
            }
            DumpTimes(std::string)
            {
            }

            template<typename, typename... Args>
            inline auto now(Args&&...) -> Ret_t
            {
            }

            template<typename... Args>
            inline auto append(Args&&...) -> void
            {
            }

            template<typename... Args>
            inline auto flush(Args&&...) -> void
            {
            }
        };

    } // namespace openPMD
} // namespace picongpu
