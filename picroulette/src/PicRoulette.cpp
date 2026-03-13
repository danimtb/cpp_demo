#include "PicRoulette.h"
#include "Config.hpp"
#include "OpenGLRenderer.h"
#include "QuickDrawRecognizer.h"
#include "VideoInput.h"
#include "hud.h"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

namespace {
enum class GameState { RUNNING, VICTORY, GAME_OVER };
}

namespace conan {
PicRoulette::PicRoulette(const Config &cfg)
    : r_config(cfg), r_roundDuration(cfg.get<int>("game.roundDuration")),
      r_maxHistory(cfg.get<int>("game.maxHistory")) {

    r_videoInput = std::make_unique<VideoInput>(cfg);
    // SDL2/OpenGL window
    r_renderer = std::make_unique<OpenGLRenderer>(cfg, r_videoInput->getFrameWidth(),
                                                  r_videoInput->getFrameHeight());

    loadLabels(assetPath(cfg.get<std::string>("recognizer.labels")));
    r_recognizer = std::make_unique<QuickDrawRecognizer>(cfg, r_labels);

    r_captureZone = std::make_unique<CaptureZone>(cfg, r_videoInput->getFrameWidth(),
                                                  r_videoInput->getFrameHeight());
}

void PicRoulette::run() {
    bool running = true;
    cv::Mat frame;

    std::chrono::steady_clock::time_point endTime{};
    const auto autoRestartTime = r_config.get<int>("game.autoRestartTime");

    auto gameState = GameState::RUNNING;

    restart();
    while (running) {
        // --- Event handling ---
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT)
                running = false;
            if (ev.type == SDL_KEYDOWN) {
                if (ev.key.keysym.sym == SDLK_ESCAPE || ev.key.keysym.sym == SDLK_q)
                    running = false;
                if (ev.key.keysym.sym == SDLK_r) {
                    restart();
                    gameState = GameState::RUNNING;
                }
            }
            if (ev.type == SDL_WINDOWEVENT && ev.window.event == SDL_WINDOWEVENT_RESIZED) {
                r_renderer->handleResize(ev.window.data1, ev.window.data2);
                r_captureZone->updateWindowSize(ev.window.data1, ev.window.data2);
            }
        }

        // --- Camera capture ---
        r_videoInput->getFrame(frame);
        if (frame.empty())
            continue;

        // --- HUD elements ---
        std::vector<conan::RenderableElement> elements;
        switch (gameState) {
        case GameState::RUNNING:
            analyzeFrame(frame);
            if (r_match) {
                gameState = GameState::VICTORY;
                endTime = std::chrono::steady_clock::now();
            } else if (isGameOver()) {
                gameState = GameState::GAME_OVER;
                endTime = std::chrono::steady_clock::now();
            } else {
                elements.push_back(TextElement{"Draw " + displayItem(),
                                               conan::TextAnchor::TOP_LEFT, 30, 20, 50,
                                               conan::Color::Green});
                elements.push_back(
                    TextElement{timeMissing(), conan::TextAnchor::TOP_RIGHT, 30, 20});
            }
            // --- Add capture zone text ---
            if (r_bestDetection != "unknown") {
                elements.push_back(
                    TextElement{"Detected: " + r_bestDetection, conan::TextAnchor::TOP_LEFT,
                                r_captureZone->getZone().x + 10,
                                r_captureZone->getZone().y + r_captureZone->getZone().height + 20,
                                44, conan::Color::Cyan});
            }
            r_renderer->renderFrame(frame, elements, *r_captureZone);
            break;
        case GameState::VICTORY:
            elements.push_back(TextElement{"VICTORY!", conan::TextAnchor::CENTER, 0, -40, 65,
                                           conan::Color::Yellow});
            elements.push_back(TextElement{"You drew " + displayItem(),
                                           conan::TextAnchor::CENTER, 0, 20, 40,
                                           conan::Color::Yellow});

            elements.push_back(ImageElement(r_winnerDrawing)
                                   .setSize(256, 256)
                                   .setAnchor(conan::TextAnchor::CENTER)
                                   .setOffset(0, 200));
            r_renderer->renderFrame(frame, elements);
            break;
        case GameState::GAME_OVER:
            const auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - endTime).count();

            elements.push_back(
                TextElement{"GAME OVER", conan::TextAnchor::CENTER, 0, 0, 64, conan::Color::Red});
            elements.push_back(
                TextElement{"Restarting in " + std::to_string(autoRestartTime - elapsed) + "...",
                            conan::TextAnchor::CENTER, 0, 60, 40, conan::Color::Pink});
            if (elapsed >= autoRestartTime) {
                restart();
                gameState = GameState::RUNNING;
            }
            r_renderer->renderFrame(frame, elements);
            break;
        }

        // --- swap buffers ---
        SDL_GL_SwapWindow(r_renderer->getWindow());
    }
}

std::string PicRoulette::displayItem() const {
    static const std::vector<std::string> vowels = {"a", "e", "i", "o", "u"};
    const auto word = r_labels[r_currentIndex];
    if (std::find(vowels.begin(), vowels.end(), std::string(1, tolower(word[0]))) != vowels.end())
        return "an " + word;
    return "a " + word;
}

// Return remaining time as mm:ss
std::string PicRoulette::timeMissing() const {
    auto now = std::chrono::steady_clock::now();
    int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - r_lastChange).count();
    int remaining = r_roundDuration - elapsed;
    if (remaining < 0)
        remaining = 0;
    int m = remaining / 60, s = remaining % 60;
    char buf[6];
    snprintf(buf, sizeof(buf), "%02d:%02d", m, s);
    return std::string(buf);
}

// Analyze a frame: paper detection, segmentation, ONNX prediction
void PicRoulette::analyzeFrame(const cv::Mat &frame) {
    // Update capture zone state
    const int winW = r_renderer->getWidth();
    const int winH = r_renderer->getHeight();
    r_captureZone->updateZoneState(frame, winW, winH);

    r_match = false;
    // Always use capture zone (no more paper detection)
    // Map capture zone to frame coordinates
    cv::Rect roi = r_captureZone->mapToFrame(frame, winW, winH);

    // Skip if ROI invalid
    if (roi.width <= 0 || roi.height <= 0) {
        r_bestDetection = "unknown";
        return;
    }

    const auto &[lastDetection, lastConfidence] =
        r_recognizer->recognize(frame, roi, r_captureZone->isActive());

    r_recentDetections.push_back(lastDetection);
    if (r_recentDetections.size() > r_maxHistory)
        r_recentDetections.pop_front();

    // count frequencies
    std::unordered_map<std::string, int> counts;
    for (auto &d : r_recentDetections)
        counts[d]++;

    auto best = std::max_element(counts.begin(), counts.end(),
                                 [](auto &a, auto &b) { return a.second < b.second; });

    r_bestDetection = best->first;
    if (lastConfidence < r_captureZone->getRecognitionThreshold())
        r_bestDetection = "unknown";

    if (r_bestDetection == currentItem()) {
        r_match = true;
        r_winnerDrawing = r_renderer->matToTexture(r_recognizer->lastPaperROI());
    }
}

// Restart / next word
void PicRoulette::restart() {
    // pick random item
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, r_labels.size() - 1);
    r_currentIndex = dis(gen);
    r_lastChange = std::chrono::steady_clock::now();
    r_match = false;
    r_bestDetection = "unknown";
    r_recentDetections.clear();
    r_renderer->releaseTexture(r_winnerDrawing);
}

bool PicRoulette::isGameOver() const {
    auto now = std::chrono::steady_clock::now();
    int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - r_lastChange).count();
    return elapsed >= r_roundDuration && !r_match;
}

void PicRoulette::loadLabels(const std::string &labelsPath) {
    std::ifstream f(labelsPath);
    if (!f)
        throw std::runtime_error("Failed to open labels file: " + labelsPath);
    nlohmann::json j;
    f >> j;
    if (!j.is_array())
        throw std::runtime_error("Labels file is not a JSON array");
    r_labels = j.get<std::vector<std::string>>();
}

}; // namespace conan
