#pragma once
#include "CaptureZone.hpp"
#include "Config.hpp"
#include "hud.h"
#include <SDL.h>
#include <SDL_opengl.h>
#include <SDL_ttf.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace conan {
struct TextTexture {
    GLuint textureID{};
    int w{}, h{};
};

class OpenGLRenderer {
  public:
    OpenGLRenderer(const Config &cfg, int w, int h);
    ~OpenGLRenderer();

    void renderFrame(const cv::Mat &frame, const std::vector<RenderableElement> &elements,
                     const std::optional<CaptureZone> &captureZone = std::nullopt);
    void updateTexture(const cv::Mat &frame);
    SDL_Window *getWindow();
    int getWidth() const;
    int getHeight() const;
    void handleResize(int newWidth, int newHeight);
    GLuint matToTexture(const cv::Mat &mat);
    void releaseTexture(GLuint &texID);

  private:
    void updateViewport();
    static Uint32 colorKey(SDL_Color c) { return (c.r << 24) | (c.g << 16) | (c.b << 8) | c.a; }
    TextTexture getTextTexture(const TextElement &element);
    GLuint loadImageTexture(const std::string &path);

    void drawText(const TextElement &elem);
    void drawText(const TextTexture &tex, int x, int y);
    std::pair<int, int> computePosition(const TextTexture &t, TextAnchor anchor, int offsetX,
                                        int offsetY);
    const cv::Mat correctFrame(const cv::Mat &frame);

    void renderQuad();
    void renderBanner(int bannerHeight = 100);

    void drawImage(const ImageElement &img);

    void drawVerticalGradient(float x, float y, float w, float h, bool fadeDown,
                              float alpha = 0.95f, float plateauFrac = 0.8f);
    void drawCaptureZone(const CaptureZone &captureZone);
    const Config &cfg;

    std::map<std::tuple<std::string, std::string, int, Uint32>, TextTexture> textCache;
    SDL_Window *window;
    SDL_GLContext glContext;
    GLuint textureID;
    GLuint r_logoTextureID;
    GLuint r_auditQrTextureID;
    GLuint r_registrationQrTextureID{};
    GLuint r_xTextureID;
    GLuint r_githubTextureID;

    std::map<std::string, std::tuple<GLuint, int, int>> r_images;
    int windowWidth;
    int windowHeight;
    float aspectRatio;
    int textureWidth;
    int textureHeight;
};

} // namespace conan
