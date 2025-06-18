/*
author : @rebwar_ai
*/
#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <string>
#include <fstream>

namespace L {
    void log(const std::string& message, const std::string& filename = "log.txt") {
        std::ofstream file(filename, std::ios::app); // Append mode

        if (file.is_open()) {
            // Optionally prepend a timestamp
            /*
            auto now = std::chrono::system_clock::now();
            std::time_t time_now = std::chrono::system_clock::to_time_t(now);
            file << std::ctime(&time_now) << ": ";
            */

            file << message;
            file.close();
        } else {
            std::cerr << "Unable to open log file: " << filename << std::endl;
        }
    }
}

#endif