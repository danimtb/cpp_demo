#pragma once
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace conan {
class Config {
  public:
    Config(int argc, char *argv[], const std::string &jsonPath) {
        std::ifstream f(jsonPath);
        if (!f)
            throw std::runtime_error("Failed to open config file: " + jsonPath);
        f >> j;

        // 2) Parse CLI arguments as --key=value
        for (auto i{1}; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg.rfind("--", 0) == 0) {
                auto eq = arg.find('=');
                if (eq == std::string::npos)
                    continue;
                const std::string key = arg.substr(2, eq - 2);
                const std::string value = arg.substr(eq + 1);
                cli[key] = value;
            }
        }
    }

    template <typename T> T get(const std::string &key, const T &defaultValue = T()) const {
        // check CLI first
        auto it = cli.find(key);
        if (it != cli.end())
            return lexicalCast<T>(it->second);

        // JSON traversal
        nlohmann::json current = j;
        size_t start = 0, dot;
        while ((dot = key.find('.', start)) != std::string::npos) {
            std::string part = key.substr(start, dot - start);
            if (!current.contains(part))
                return defaultValue;
            current = current[part];
            start = dot + 1;
        }
        std::string last = key.substr(start);
        if (!current.contains(last))
            return defaultValue;
        return current.at(last).get<T>();
    }

  private:
    nlohmann::json j;
    std::unordered_map<std::string, std::string> cli;

    static int lexicalCastImpl(const std::string &s, int) { return std::stoi(s); }
    static float lexicalCastImpl(const std::string &s, float) { return std::stof(s); }
    static double lexicalCastImpl(const std::string &s, double) { return std::stod(s); }
    static bool lexicalCastImpl(const std::string &s, bool) {
        return s == "1" || s == "true" || s == "yes";
    }
    static std::string lexicalCastImpl(const std::string &s, std::string) { return s; }

    template <typename T>
    static T lexicalCast(const std::string &s) {
        return lexicalCastImpl(s, T{});
    }
};
} // namespace conan
