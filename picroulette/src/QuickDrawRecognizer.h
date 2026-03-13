#pragma once
#include "Config.hpp"
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace conan {

// Preprocessing configuration constants
struct PreprocessingConfig {
    // Image scaling and resizing (used in scaleToTargetWidth)
    static constexpr int TARGET_WIDTH = 512; // Target width for image scaling before processing

    // Adaptive threshold parameters (used in createAdaptiveThreshold)
    static constexpr int BLOCK_SIZE = 31; // Block size for adaptive threshold (must be odd)
    static constexpr int BIAS = 7;        // Bias for adaptive threshold (higher = more foreground)

    // Component filtering and selection (used in filterAndUnionComponents)
    static constexpr int MAX_COMPONENTS = 8; // Maximum number of connected components to keep

    // Paper region detection thresholds (used in extractPaperRegion)
    static constexpr double MIN_RED_AREA =
        5000.0; // Minimum area for red portfolio detection (HSV filtering)
    static constexpr double MIN_WHITE_AREA =
        2000.0; // Minimum area for white paper detection inside red region
    static constexpr int MIN_BOX_SIZE = 50; // Minimum bounding box size for valid regions

    // Aspect ratio validation (used in extractPaperRegion)
    static constexpr double MIN_ASPECT_RATIO =
        0.5; // Minimum valid aspect ratio for paper detection
    static constexpr double MAX_ASPECT_RATIO =
        2.0; // Maximum valid aspect ratio for paper detection

    // Morphological operations (used in createAdaptiveThreshold and filterAndUnionComponents)
    static constexpr int MORPH_KERNEL_SIZE =
        1; // Kernel size for morphological cleanup (open/close)
    static constexpr int MORPH_CLOSE_SIZE =
        5; // Kernel size for closing operation (join nearby strokes)
};

class QuickDrawRecognizer {
  public:
    QuickDrawRecognizer(const Config &cfg, const std::vector<std::string> &labels);
    std::pair<std::string, float> recognize(const cv::Mat &input, const cv::Rect &captureZone,
                                            bool zoneActive);

    const std::string &lastLabel() const { return m_lastLabel; }
    float lastConfidence() const { return m_lastConf; }
    cv::Mat lastPaperROI() const { return r_lastPaperROI; }

  private:
    cv::Mat extractFromCaptureZone(const cv::Mat &frame, const cv::Rect &captureZone);
    cv::Mat preprocess(const cv::Mat &imgGray, int thicken, double pad);

    // Helper functions for preprocessing
    cv::Mat scaleToTargetWidth(const cv::Mat &gray);
    cv::Mat createAdaptiveThreshold(const cv::Mat &gray);
    cv::Mat filterAndUnionComponents(const cv::Mat &mask, const cv::Mat &scaled_gray);
    cv::Rect calculateBoundingBox(const cv::Mat &keep, const cv::Mat &scaled_gray);
    cv::Mat cropImage(const cv::Mat &gray, const cv::Rect &box);
    cv::Mat finalizeImage(const cv::Mat &img, int thicken, double pad);
    void applySoftmax(std::vector<float> &logits);

    // Debug helper functions
    void showDebugWindow(const std::string &windowName, const cv::Mat &image);
    void showDebugBoundingBox(const cv::Mat &image, const cv::Rect &box, const std::string &step);

    torch::Tensor toTensorNCHW(const cv::Mat &input);

    const Config &r_config;
    torch::jit::script::Module r_module;
    torch::Device r_device{torch::kCPU};
    std::vector<std::string> r_labels;

    std::string m_lastLabel;
    cv::Mat r_lastPaperROI;
    float m_lastConf = 0.0f;
};
} // namespace conan
