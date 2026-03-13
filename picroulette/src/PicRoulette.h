#pragma once
#include "CaptureZone.hpp"
#include "Config.hpp"
#include "OpenGLRenderer.h"
#include "QuickDrawRecognizer.h"
#include "VideoInput.h"
#include <SDL_opengl.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace conan {
class PicRoulette {
  public:
    PicRoulette(const Config &cfg);
    void run();

  private:
    void loadLabels(const std::string &labelsPath);
    std::string currentItem() const { return r_labels[r_currentIndex]; }
    std::string displayItem() const;

    // Return remaining time as mm:ss
    std::string timeMissing() const;

    // Analyze a frame: paper detection, segmentation, ONNX prediction
    void analyzeFrame(const cv::Mat &frame);

    // Restart / next word
    void restart();

    bool isGameOver() const;

    const Config &r_config;
    std::unique_ptr<CaptureZone> r_captureZone;
    std::unique_ptr<VideoInput> r_videoInput;
    std::unique_ptr<OpenGLRenderer> r_renderer;

    std::vector<std::string> r_labels;
    std::string r_bestDetection{};
    std::deque<std::string> r_recentDetections;
    int r_currentIndex{};
    int r_roundDuration{};
    int r_maxHistory{};
    GLuint r_winnerDrawing{};
    std::chrono::steady_clock::time_point r_lastChange;
    bool r_match{};
    std::unique_ptr<QuickDrawRecognizer> r_recognizer;
};
} // namespace conan
