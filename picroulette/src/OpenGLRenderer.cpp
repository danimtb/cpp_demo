#include "OpenGLRenderer.h"
#include "CaptureZone.hpp"
#include "Config.hpp"
#include "utils.hpp"
#include <SDL_image.h>
#include <SDL_opengl.h>
#include <optional>
#include <string>

namespace conan {
using namespace cv;

OpenGLRenderer::OpenGLRenderer(const Config &cfg, int initialWidth, int initialHeight)
    : aspectRatio(static_cast<float>(initialWidth) / initialHeight), windowWidth(initialWidth),
      windowHeight(initialHeight), textureWidth(initialWidth), textureHeight(initialHeight),
      cfg(cfg) {
    window =
        SDL_CreateWindow("Multipose Suit Overlay", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                         windowWidth, windowHeight, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window)
        throw std::runtime_error(std::string("SDL_CreateWindow Error: ") + SDL_GetError());
    glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        SDL_DestroyWindow(window);
        throw std::runtime_error(std::string("SDL_GL_CreateContext Error: ") + SDL_GetError());
    }
    updateViewport();
    glEnable(GL_TEXTURE_2D);
    glClearColor(0, 0, 0, 1);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 NULL);

    // Setup to render correctly the TTF fonts
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Load log
    r_logoTextureID = loadImageTexture(assetPath("images/conan2-logo.svg"));
    r_xTextureID = loadImageTexture(assetPath("images/x-logo.svg"));
    r_githubTextureID = loadImageTexture(assetPath("images/github-logo.svg"));
    if (const auto auditQR = cfg.get<std::string>("ui.auditQR"); auditQR != "") {
        r_auditQrTextureID = loadImageTexture(assetPath(auditQR));
    }
    if (const auto registrationQR = cfg.get<std::string>("ui.registrationQR");
        registrationQR != "") {
        r_registrationQrTextureID = loadImageTexture(assetPath(registrationQR));
    }
}

OpenGLRenderer::~OpenGLRenderer() {
    glDeleteTextures(1, &textureID);
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void OpenGLRenderer::handleResize(int newWidth, int newHeight) {
    if (newWidth / float(newHeight) > aspectRatio)
        newWidth = int(newHeight * aspectRatio);
    else
        newHeight = int(newWidth / aspectRatio);

    // enforce corrected size
    SDL_SetWindowSize(window, newWidth, newHeight);
    updateViewport();
}

GLuint OpenGLRenderer::matToTexture(const cv::Mat &mat) {
    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);

    GLenum format = (mat.channels() == 1) ? GL_LUMINANCE : GL_BGR;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, format, GL_UNSIGNED_BYTE,
                 mat.data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    return texID;
}

void OpenGLRenderer::releaseTexture(GLuint &texID) {
    if (texID != 0) {
        glDeleteTextures(1, &texID);
        texID = 0;
    }
}

void OpenGLRenderer::updateViewport() {
    SDL_GetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, windowHeight, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void OpenGLRenderer::updateTexture(const Mat &frame) {
    glBindTexture(GL_TEXTURE_2D, textureID);
    if (frame.cols != textureWidth || frame.rows != textureHeight) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE,
                     frame.data);
        textureWidth = frame.cols;
        textureHeight = frame.rows;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE,
                        frame.data);
    }
    glFlush();
}

void OpenGLRenderer::renderQuad() {
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(windowWidth, 0);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(windowWidth, windowHeight);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, windowHeight);
    glEnd();
}

SDL_Window *OpenGLRenderer::getWindow() { return window; }

int OpenGLRenderer::getWidth() const { return windowWidth; }

int OpenGLRenderer::getHeight() const { return windowHeight; }

std::pair<int, int> OpenGLRenderer::computePosition(const TextTexture &t, TextAnchor anchor,
                                                    int offsetX, int offsetY) {
    int x = 0, y = 0;
    switch (anchor) {
    case TextAnchor::TOP_LEFT:
        x = offsetX;
        y = offsetY;
        break;
    case TextAnchor::TOP_RIGHT:
        x = windowWidth - t.w - offsetX;
        y = offsetY;
        break;
    case TextAnchor::BOTTOM_LEFT:
        x = offsetX;
        y = windowHeight - t.h - offsetY;
        break;
    case TextAnchor::BOTTOM_RIGHT:
        x = windowWidth - t.w - offsetX;
        y = windowHeight - t.h - offsetY;
        break;
    case TextAnchor::CENTER:
        x = (windowWidth - t.w) / 2 + offsetX;
        y = (windowHeight - t.h) / 2 + offsetY;
        break;
    }
    return {x, y};
}

TextTexture OpenGLRenderer::getTextTexture(const TextElement &element) {
    const auto fontPath =
        (std::filesystem::path(assetPath("fonts")) / element.font).string() + ".ttf";
    const auto color = getColor(element.color);
    auto key = std::make_tuple(element.text, fontPath, element.fontSize, colorKey(color));

    // if exists → reuse
    auto it = textCache.find(key);
    if (it != textCache.end()) {
        return it->second;
    }

    // else → create new
    TTF_Font *font = TTF_OpenFont(fontPath.c_str(), element.fontSize);
    if (!font)
        throw std::runtime_error(std::string("TTF_OpenFont error: ") + TTF_GetError());

    SDL_Surface *surf = TTF_RenderUTF8_Blended(font, element.text.c_str(), color);
    if (!surf) {
        TTF_CloseFont(font);
        throw std::runtime_error(std::string("TTF_RenderUTF8_Blended error: ") + TTF_GetError());
    }
    // Convert surface to RGBA8888 (safe for OpenGL)
    SDL_Surface *conv = SDL_ConvertSurfaceFormat(surf, SDL_PIXELFORMAT_ABGR8888, 0);
    SDL_FreeSurface(surf);
    surf = conv;
    GLuint texId;
    // if (!surf)
    //     return result;

    // Generate OpenGL texture
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload pixels to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf->w, surf->h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 surf->pixels);

    int textW, textH;
    if (TTF_SizeUTF8(font, element.text.c_str(), &textW, &textH) != 0) {
        std::cerr << "TTF_SizeUTF8 error: " << TTF_GetError() << std::endl;
    }

    // glGenTextures(1, &texId);
    // glBindTexture(GL_TEXTURE_2D, texId);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf->w, surf->h, 0,
    //              GL_BGRA, GL_UNSIGNED_BYTE, surf->pixels);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    TextTexture t{texId, surf->w, surf->h};
    textCache[key] = t;

    SDL_FreeSurface(surf);
    TTF_CloseFont(font);

    return t;
}

void OpenGLRenderer::drawText(const TextElement &elem) {
    TextTexture tex = getTextTexture(elem);
    auto [x, y] = computePosition(tex, elem.anchor, elem.offsetX, elem.offsetY);
    drawText(tex, x, y);
}

void OpenGLRenderer::drawText(const TextTexture &tex, int x, int y) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex.textureID);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(x, y);
    glTexCoord2f(1, 0);
    glVertex2f(x + tex.w, y);
    glTexCoord2f(1, 1);
    glVertex2f(x + tex.w, y + tex.h);
    glTexCoord2f(0, 1);
    glVertex2f(x, y + tex.h);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}

const cv::Mat OpenGLRenderer::correctFrame(const cv::Mat &frame) {
    cv::Mat flipped;
    cv::flip(frame, flipped, 1);
    cv::Mat rgb;
    cv::cvtColor(flipped, rgb, cv::COLOR_BGR2RGB);
    cv::Mat disp;
    cv::resize(rgb, disp, cv::Size(getWidth(), getHeight()));
    return disp;
}

GLuint OpenGLRenderer::loadImageTexture(const std::string &path) { //, int width, int height) {
    // Load image with SDL_image
    SDL_Surface *surf = IMG_Load(path.c_str());
    if (!surf) {
        throw std::runtime_error("Failed to load image: " + path + " -> " + IMG_GetError());
    }

    // Determine the pixel format
    GLenum format = GL_RGB;
    if (surf->format->BytesPerPixel == 4) {
        format = GL_RGBA;
    } else if (surf->format->BytesPerPixel != 3) {
        SDL_FreeSurface(surf);
        throw std::runtime_error("Unsupported image format");
    }

    // Generate OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, format, surf->w, surf->h, 0, format, GL_UNSIGNED_BYTE,
                 surf->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    SDL_FreeSurface(surf);

    // r_images[name] = {textureID, surf->w, surf->h};
    return textureID;
}

void OpenGLRenderer::drawVerticalGradient(float x, float y, float w, float h, bool fadeDown,
                                          float alpha, float plateauFrac) {
    // Save GL state
    GLboolean wasTex = glIsEnabled(GL_TEXTURE_2D);
    GLboolean wasBlend = glIsEnabled(GL_BLEND);
    GLint prevShade;
    glGetIntegerv(GL_SHADE_MODEL, &prevShade);
    GLint prevSrc, prevDst;
    glGetIntegerv(GL_BLEND_SRC, &prevSrc);
    glGetIntegerv(GL_BLEND_DST, &prevDst);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);

    float plateauH = h * plateauFrac;

    if (fadeDown) {
        // Top gradient (opaque → transparent)
        float yOpaque = y;
        float yPlateau = y + plateauH;
        float yFadeEnd = y + h;

        // 1) Plateau band
        glBegin(GL_QUADS);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x, yOpaque);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x + w, yOpaque);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x + w, yPlateau);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x, yPlateau);
        glEnd();

        // 2) Fade to transparent
        glBegin(GL_QUADS);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x, yPlateau);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x + w, yPlateau);
        glColor4f(0.f, 0.f, 0.f, 0.f);
        glVertex2f(x + w, yFadeEnd);
        glColor4f(0.f, 0.f, 0.f, 0.f);
        glVertex2f(x, yFadeEnd);
        glEnd();
    } else {
        // Bottom gradient (transparent → opaque)
        float yFadeStart = y;
        float yPlateau = y + (h - plateauH);
        float yOpaque = y + h;

        // 1) Fade region
        glBegin(GL_QUADS);
        glColor4f(0.f, 0.f, 0.f, 0.f);
        glVertex2f(x, yFadeStart);
        glColor4f(0.f, 0.f, 0.f, 0.f);
        glVertex2f(x + w, yFadeStart);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x + w, yPlateau);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x, yPlateau);
        glEnd();

        // 2) Plateau band
        glBegin(GL_QUADS);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x, yPlateau);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x + w, yPlateau);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x + w, yOpaque);
        glColor4f(0.f, 0.f, 0.f, alpha);
        glVertex2f(x, yOpaque);
        glEnd();
    }

    // Restore GL state
    if (wasTex)
        glEnable(GL_TEXTURE_2D);
    else
        glDisable(GL_TEXTURE_2D);
    if (wasBlend)
        glEnable(GL_BLEND);
    else
        glDisable(GL_BLEND);
    glBlendFunc(prevSrc, prevDst);
    glShadeModel(prevShade);
    glColor4f(1.f, 1.f, 1.f, 1.f);
}
void OpenGLRenderer::renderBanner(int bannerHeight) {
    int winW = windowWidth;
    int winH = windowHeight;

    // --- TOP GRADIENT (opaque → transparent downwards) ---
    drawVerticalGradient(0, 0, winW, bannerHeight, true);

    // --- BOTTOM GRADIENT (transparent → opaque upwards) ---
    drawVerticalGradient(0, winH - bannerHeight, winW, bannerHeight, false);

    // 2) Draw centered logo
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, r_logoTextureID);
    int logoW = 460, logoH = 80;
    float xLogo = (winW - logoW) / 2.0f;
    float yLogo = winH - bannerHeight + (bannerHeight - logoH) / 2.0f;
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(xLogo, yLogo);
    glTexCoord2f(1, 0);
    glVertex2f(xLogo + logoW, yLogo);
    glTexCoord2f(1, 1);
    glVertex2f(xLogo + logoW, yLogo + logoH);
    glTexCoord2f(0, 1);
    glVertex2f(xLogo, yLogo + logoH);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    // 3) Meeting text left
    const auto meet = cfg.get<std::string>("ui.meet", "Meeting C++");
    TextTexture meetTex = getTextTexture(TextElement{meet});
    drawText(meetTex, 40, yLogo + (logoH - meetTex.h) / 2);

    // 4) Year text right
    const auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm *localTime = std::localtime(&t);
    int year = 1900 + localTime->tm_year;
    TextTexture yearTex = getTextTexture(TextElement{std::to_string(year)});
    drawText(yearTex, winW - 40 - yearTex.w, yLogo + (logoH - yearTex.h) / 2);

    // 5) QR codes (left and right bottom corners, above banner)
    int qrSize = 160;
    int margin = 50;
    auto drawQR = [&](GLuint texId, int x, int y, const std::string &label) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texId);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(x, y);
        glTexCoord2f(1, 0);
        glVertex2f(x + qrSize, y);
        glTexCoord2f(1, 1);
        glVertex2f(x + qrSize, y + qrSize);
        glTexCoord2f(0, 1);
        glVertex2f(x, y + qrSize);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        TextTexture textTex = getTextTexture(TextElement{label}.setFontSize(20));
        drawText(textTex, x + (qrSize - textTex.w) / 2, y + qrSize + 11);
    };

    if (r_auditQrTextureID != 0)
        drawQR(r_auditQrTextureID, margin, winH - bannerHeight - qrSize - margin, "Register");
    if (r_registrationQrTextureID != 0)
        drawQR(r_registrationQrTextureID, winW - margin - qrSize,
               winH - bannerHeight - qrSize - margin, "Register");

    int iconSize = 30;
    int spacing = 40;
    auto drawIconWithText = [&](GLuint texId, float x, float y, const std::string &label) {
        // Draw icon
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texId);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(x, y);
        glTexCoord2f(1, 0);
        glVertex2f(x + iconSize, y);
        glTexCoord2f(1, 1);
        glVertex2f(x + iconSize, y + iconSize);
        glTexCoord2f(0, 1);
        glVertex2f(x, y + iconSize);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        // Draw label to the right of the icon
        TextTexture textTex = getTextTexture(TextElement{label}.setFontSize(20));
        int textX = static_cast<int>(x + iconSize + 5);
        int textY = static_cast<int>(y + (iconSize - textTex.h) / 2);
        drawText(textTex, textX, textY);
    };
    // Social media icons + labels
    float xIcon = xLogo + logoW + 100;
    float yIcon = yLogo + (logoH - iconSize) / 2;

    drawIconWithText(r_xTextureID, xIcon, yIcon, "@conan_io");
    xIcon += iconSize + spacing + 120;
    drawIconWithText(r_githubTextureID, xIcon, yIcon, "conan-io");
}

void OpenGLRenderer::renderFrame(const cv::Mat &frame,
                                 const std::vector<RenderableElement> &elements,
                                 const std::optional<CaptureZone> &captureZone) {
    updateTexture(correctFrame(frame));
    renderQuad();

    for (const auto &elem : elements) {
        std::visit(
            [this](auto &&e) {
                using T = std::decay_t<decltype(e)>;
                if constexpr (std::is_same_v<T, ImageElement>)
                    drawImage(e);
            },
            elem);
    }
    if (captureZone) {
        // Draw capture zone
        drawCaptureZone(captureZone.value());
    }
    // Render banner should be at the end because if not, the drawImage will blank the screen
    renderBanner(100);
    for (const auto &elem : elements) {
        std::visit(
            [this](auto &&e) {
                using T = std::decay_t<decltype(e)>;
                if constexpr (std::is_same_v<T, TextElement>)
                    drawText(e);
            },
            elem);
    }
}

void OpenGLRenderer::drawImage(const ImageElement &img) {
    int x = img.offsetX;
    int y = img.offsetY;

    // adjust according to anchor
    switch (img.anchor) {
    case TextAnchor::TOP_LEFT:
        break;
    case TextAnchor::TOP_RIGHT:
        x = windowWidth - img.width - img.offsetX;
        break;
    case TextAnchor::BOTTOM_LEFT:
        y = windowHeight - img.height - img.offsetY;
        break;
    case TextAnchor::BOTTOM_RIGHT:
        x = windowWidth - img.width - img.offsetX;
        y = windowHeight - img.height - img.offsetY;
        break;
    case TextAnchor::CENTER:
        x = (windowWidth - img.width) / 2 + img.offsetX;
        y = (windowHeight - img.height) / 2 + img.offsetY;
        break;
    }

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, img.textureId);
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_QUADS);
    glTexCoord2f(0.f, 0.f);
    glVertex2f(x, y);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(x + img.width, y);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(x + img.width, y + img.height);
    glTexCoord2f(0.f, 1.f);
    glVertex2f(x, y + img.height);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void OpenGLRenderer::drawCaptureZone(const CaptureZone &captureZone) {
    const auto &zone = captureZone.getZone();
    if (zone.width <= 0 || zone.height <= 0)
        return;

    // Save current OpenGL state
    GLboolean textureEnabled = glIsEnabled(GL_TEXTURE_2D);
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);
    GLfloat currentColor[4];
    glGetFloatv(GL_CURRENT_COLOR, currentColor);

    // Set up OpenGL for 2D drawing
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Choose color based on state
    if (captureZone.isActive()) {
        glColor4f(0.0f, 1.0f, 0.0f, 0.8f); // Green with transparency
    } else {
        glColor4f(1.0f, 0.0f, 0.0f, 0.8f); // Red with transparency
    }

    // Draw rectangle outline
    glLineWidth(3.0f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(zone.x, zone.y);
    glVertex2f(zone.x + zone.width, zone.y);
    glVertex2f(zone.x + zone.width, zone.y + zone.height);
    glVertex2f(zone.x, zone.y + zone.height);
    glEnd();

    // Draw semi-transparent fill
    glColor4f(0.0f, 0.0f, 0.0f, 0.3f);
    glBegin(GL_QUADS);
    glVertex2f(zone.x, zone.y);
    glVertex2f(zone.x + zone.width, zone.y);
    glVertex2f(zone.x + zone.width, zone.y + zone.height);
    glVertex2f(zone.x, zone.y + zone.height);
    glEnd();

    // Restore OpenGL state completely
    if (textureEnabled) {
        glEnable(GL_TEXTURE_2D);
    } else {
        glDisable(GL_TEXTURE_2D);
    }

    if (!blendEnabled) {
        glDisable(GL_BLEND);
    }

    // Restore original color
    glColor4f(currentColor[0], currentColor[1], currentColor[2], currentColor[3]);
}
} // namespace conan
