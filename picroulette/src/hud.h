#pragma once
#include <SDL.h>
#include <array>
#include <string>
#include <variant>
#include <SDL_opengl.h>

namespace conan {
enum class TextAnchor { TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER };

enum class Color {
    White,
    Black,
    Red,
    Green,
    Blue,
    Yellow,
    Cyan,
    Magenta,
    Gray,
    Orange,
    Purple,
    Pink,
    Brown,

    Count // sentinel for size
};

constexpr std::array<SDL_Color, static_cast<size_t>(Color::Count)> ColorTable{{
    SDL_Color{255, 255, 255, 255}, // White
    SDL_Color{0, 0, 0, 255},       // Black
    SDL_Color{255, 0, 0, 255},     // Red
    SDL_Color{0, 255, 0, 255},     // Green
    SDL_Color{0, 0, 255, 255},     // Blue
    SDL_Color{255, 255, 0, 255},   // Yellow
    SDL_Color{0, 255, 255, 255},   // Cyan
    SDL_Color{255, 0, 255, 255},   // Magenta
    SDL_Color{128, 128, 128, 255}, // Gray
    SDL_Color{255, 165, 0, 255},   // Orange
    SDL_Color{128, 0, 128, 255},   // Purple
    SDL_Color{255, 192, 203, 255}, // Pink
    SDL_Color{139, 69, 19, 255},   // Brown
}};

constexpr SDL_Color getColor(Color c) { return ColorTable[static_cast<size_t>(c)]; }

struct TextElement {
    std::string text;
    TextAnchor anchor = TextAnchor::TOP_LEFT;
    int offsetX = 0;
    int offsetY = 0;
    int fontSize = 40;
    Color color = Color::White;
    std::string font = "HackNerdFont-Bold";

    TextElement& setFontSize(int size) {
        fontSize = size;
        return *this;
    }
    TextElement& setColor(Color c) {
        color = c;
        return *this;
    }
    TextElement& setFont(const std::string& fontName) {
        font = fontName;
        return *this;
    }
};

struct ImageElement {
    GLuint textureId;   // already loaded GL texture
    int width;          // desired draw size
    int height;
    int offsetX = 0;
    int offsetY = 0;
    TextAnchor anchor = TextAnchor::TOP_LEFT;

    ImageElement& setSize(int w, int h) {
        width = w; height = h; return *this;
    }
    ImageElement& setOffset(int x, int y) {
        offsetX = x; offsetY = y; return *this;
    }
    ImageElement& setAnchor(TextAnchor a) {
        anchor = a; return *this;
    }
};

using RenderableElement = std::variant<TextElement, ImageElement>;

} // namespace conan
