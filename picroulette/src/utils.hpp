#pragma once

#include <filesystem>
#include <string_view>

namespace conan {
inline auto assetPath(const std::string_view asset) -> std::string {
    static const std::filesystem::path base = ASSETS_DIR;
    return (base / asset).string();
}
} // namespace conan
