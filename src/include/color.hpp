#ifndef SANAE_COLOR_HPP
#define SANAE_COLOR_HPP

#include <cstdint>
#include <string>
#include <sstream>

class FontColor {
public:
    static constexpr const char* CLEAR  = "\033[0m";
    static constexpr const char* BLACK  = "\033[30m";
    static constexpr const char* RED    = "\033[31m";
    static constexpr const char* GREEN  = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE   = "\033[34m";
    static constexpr const char* PURPLE = "\033[35m";
    static constexpr const char* CYAN   = "\033[36m";
    static constexpr const char* WHITE  = "\033[37m";

    static std::string RGB(uint8_t R, uint8_t G, uint8_t B){
        std::stringstream buf;
        buf << "\033[38;2;" << std::to_string(R) << ";" << std::to_string(G) << ";" << std::to_string(B) << "m";

        return buf.str();
    }
};

class BgColor {
public:
    static constexpr const char* CLEAR  = "\033[0m";
    static constexpr const char* BLACK  = "\033[40m";
    static constexpr const char* RED    = "\033[41m";
    static constexpr const char* GREEN  = "\033[42m";
    static constexpr const char* YELLOW = "\033[43m";
    static constexpr const char* BLUE   = "\033[44m";
    static constexpr const char* PURPLE = "\033[45m";
    static constexpr const char* CYAN   = "\033[46m";
    static constexpr const char* WHITE  = "\033[47m";

    static std::string RGB(uint8_t R, uint8_t G, uint8_t B){
        std::stringstream buf;
        buf << "\033[48;2;" << std::to_string(R) << ";" << std::to_string(G) << ";" << std::to_string(B) << "m";

        return buf.str();
    }
};

#endif // SANAE_COLOR_HPP