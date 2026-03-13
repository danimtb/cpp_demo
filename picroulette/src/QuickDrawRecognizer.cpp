#include "QuickDrawRecognizer.h"
#include "Config.hpp"
#include "utils.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace conan {

QuickDrawRecognizer::QuickDrawRecognizer(const Config &cfg, const std::vector<std::string> &labels)
    : r_config(cfg) {
    const auto modelPath = assetPath(cfg.get<std::string>("recognizer.model"));
    try {
        r_module = torch::jit::load(modelPath);
    } catch (const c10::Error &e) {
        throw std::runtime_error("Failed to load LibTorch model from " + modelPath + ": " + e.msg());
    }
    // Setup device (runtime check for CUDA)
    try {
        torch::Tensor dummy = torch::empty({1}, torch::Device(torch::kCUDA));
        r_device = torch::Device(torch::kCUDA);
        r_module.to(r_device);
        std::cout << "Successfully connected to GPU. Running on CUDA." << std::endl;
    } catch (const c10::Error &e) {
        std::cout << "CUDA not available or LibTorch is CPU-only. Defaulting to CPU." << std::endl;
    }
    r_module.eval();
    r_labels = labels;
}

// Letterbox to square with constant background
static cv::Mat letterboxSquare(const cv::Mat &gray, int fill = 255, double padding = 0.0) {
    int h = gray.rows, w = gray.cols;
    int side = std::max(h, w);

    // Add padding to the side
    int paddingPixels = static_cast<int>(side * padding);
    int paddedSide = side + 2 * paddingPixels;

    cv::Mat canvas(paddedSide, paddedSide, CV_8U, cv::Scalar(fill));
    int y0 = (paddedSide - h) / 2;
    int x0 = (paddedSide - w) / 2;
    gray.copyTo(canvas(cv::Rect(x0, y0, w, h)));
    return canvas;
}

cv::Mat QuickDrawRecognizer::preprocess(const cv::Mat &gray, int thicken, double pad) {
    // Expect grayscale 8-bit
    CV_Assert(!gray.empty());
    CV_Assert(gray.type() == CV_8UC1);

    // 1) Scale to target width
    cv::Mat scaled_gray = scaleToTargetWidth(gray);

    // 2) Create adaptive threshold mask
    cv::Mat mask = createAdaptiveThreshold(scaled_gray);

    // 3) Filter and union components
    // cv::Mat keep = filterAndUnionComponents(mask, scaled_gray);

    // 4) Calculate bounding box
    cv::Rect box = calculateBoundingBox(mask, scaled_gray);

    // 5) Crop image
    cv::Mat cropped = cropImage(scaled_gray, box);

    // 6) Finalize image (threshold, letterbox, thicken, resize, normalize)
    return finalizeImage(cropped, thicken, pad);
}

// Convert preprocessed cv::Mat (1 channel float, HxW) to LibTorch tensor NCHW (1, 1, H, W)
torch::Tensor QuickDrawRecognizer::toTensorNCHW(const cv::Mat &input) {
    CV_Assert(input.type() == CV_32F && input.channels() == 1);
    const int height = input.rows;
    const int width = input.cols;
    torch::Tensor tensor = torch::empty({1, 1, height, width}, torch::kFloat32);
    std::memcpy(tensor.data_ptr<float>(), input.ptr<float>(0),
                static_cast<size_t>(height * width) * sizeof(float));
    return tensor.to(r_device);
}

std::pair<std::string, float>
QuickDrawRecognizer::recognize(const cv::Mat &input, const cv::Rect &captureZone, bool zoneActive) {
    cv::Mat paperROI;
    if (r_config.get<bool>("recognizer.useMockImage", false)) {
        // Load mock directly in grayscale
        paperROI = cv::imread(assetPath(r_config.get<std::string>("recognizer.imageMock")),
                              cv::IMREAD_GRAYSCALE);
        if (paperROI.empty())
            throw std::runtime_error("Failed to load paper image");
        r_lastPaperROI = paperROI.clone();
    } else {
        // Always use capture zone (no more paper detection)
        paperROI = extractFromCaptureZone(input, captureZone);

        if (paperROI.empty()) {
            return {"unknown", 0.0f};
        }
        if (paperROI.channels() != 1) {
            cv::Mat gray;
            cv::cvtColor(paperROI, gray, cv::COLOR_BGR2GRAY);
            paperROI = gray;
        }
    }

    // Preprocess → tensor
    const auto preprocessed = preprocess(paperROI, r_config.get<int>("recognizer.thicken", 0),
                                         r_config.get<double>("recognizer.pad", 0.12));
    torch::Tensor inputTensor = toTensorNCHW(preprocessed);

    // Run LibTorch inference (no grad) on r_device (CPU or CUDA)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);
    torch::NoGradGuard no_grad;
    torch::Tensor outputTensor = r_module.forward(inputs).toTensor();

    // Output is logits [1, num_classes]; move to CPU for reading if on CUDA
    outputTensor = outputTensor.squeeze(0).cpu();
    const int64_t nClasses = outputTensor.size(0);
    std::vector<float> probs(nClasses);
    const float *scores = outputTensor.data_ptr<float>();
    probs.assign(scores, scores + nClasses);
    applySoftmax(probs);

    // Top-1
    int best = int(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));
    m_lastLabel =
        (best < static_cast<int>(r_labels.size())) ? r_labels[best] : std::to_string(best);
    m_lastConf = probs[best];

    return {m_lastLabel, m_lastConf};
}

cv::Mat QuickDrawRecognizer::extractFromCaptureZone(const cv::Mat &frame,
                                                    const cv::Rect &captureZone) {
    // Ensure the zone is within frame bounds
    cv::Rect zone = captureZone;
    zone &= cv::Rect(0, 0, frame.cols, frame.rows);

    if (zone.width <= 0 || zone.height <= 0) {
        return cv::Mat();
    }

    // Extract the region from the capture zone
    cv::Mat roi = frame(zone);
    r_lastPaperROI = roi.clone();

    // Convert to grayscale if needed
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }

    showDebugWindow("Capture Zone ROI", gray);

    return gray;
}

// Helper function implementations
cv::Mat QuickDrawRecognizer::scaleToTargetWidth(const cv::Mat &gray) {
    if (gray.cols == PreprocessingConfig::TARGET_WIDTH) {
        showDebugWindow("1. Original Image", gray);
        return gray.clone();
    }

    float ratio = static_cast<float>(PreprocessingConfig::TARGET_WIDTH) / gray.cols;
    int targetHeight = static_cast<int>(gray.rows * ratio);
    cv::Mat result;
    cv::resize(gray, result, cv::Size(PreprocessingConfig::TARGET_WIDTH, targetHeight), 0, 0,
               cv::INTER_AREA);

    showDebugWindow("1. Scaled Image", result);
    return result;
}

cv::Mat QuickDrawRecognizer::createAdaptiveThreshold(const cv::Mat &gray) {
    cv::Mat mask;
    cv::adaptiveThreshold(gray, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                          PreprocessingConfig::BLOCK_SIZE, PreprocessingConfig::BIAS);

    showDebugWindow("2. Adaptive Threshold", mask);

    // Optional cleanup
    cv::morphologyEx(
        mask, mask, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {PreprocessingConfig::MORPH_KERNEL_SIZE,
                                                      PreprocessingConfig::MORPH_KERNEL_SIZE}));
    cv::morphologyEx(
        mask, mask, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {PreprocessingConfig::MORPH_KERNEL_SIZE,
                                                      PreprocessingConfig::MORPH_KERNEL_SIZE}));

    showDebugWindow("3. Morphological Cleanup", mask);

    return mask;
}

cv::Mat QuickDrawRecognizer::filterAndUnionComponents(const cv::Mat &mask,
                                                      const cv::Mat &scaled_gray) {
    cv::Mat labels, stats, cents;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, cents, 8, CV_32S);

    // Collect valid components
    struct Comp {
        int id;
        int area;
    };
    std::vector<Comp> comps;
    comps.reserve(std::max(0, n - 1));
    int imgArea = scaled_gray.rows * scaled_gray.cols;
    int minArea = std::max(150, imgArea / 2000);

    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= minArea)
            comps.push_back({i, area});
    }

    // If nothing passes the filter, use the largest raw component
    if (comps.empty() && n > 1) {
        int best = 1, bestArea = stats.at<int>(1, cv::CC_STAT_AREA);
        for (int i = 2; i < n; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > bestArea) {
                best = i;
                bestArea = area;
            }
        }
        comps.push_back({best, bestArea});
    }

    // Sort by area descending
    std::sort(comps.begin(), comps.end(),
              [](const Comp &a, const Comp &b) { return a.area > b.area; });

    // Create union mask with the largest components
    cv::Mat keep = cv::Mat::zeros(mask.size(), CV_8U);
    int added = 0;
    for (const auto &c : comps) {
        keep.setTo(255, labels == c.id);
        if (++added >= PreprocessingConfig::MAX_COMPONENTS)
            break;
    }

    showDebugWindow("4. Component Filtering", keep);

    // Join nearby strokes to avoid loose pieces
    cv::morphologyEx(
        keep, keep, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {PreprocessingConfig::MORPH_CLOSE_SIZE,
                                                      PreprocessingConfig::MORPH_CLOSE_SIZE}));

    showDebugWindow("5. Union Mask", keep);

    return keep;
}

cv::Rect QuickDrawRecognizer::calculateBoundingBox(const cv::Mat &keep,
                                                   const cv::Mat &scaled_gray) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(keep, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect box(0, 0, scaled_gray.cols, scaled_gray.rows);

    if (!contours.empty()) {
        // Find the largest contour by area
        double maxArea = 0;
        int largestContourIdx = -1;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                largestContourIdx = static_cast<int>(i);
            }
        }

        if (largestContourIdx >= 0) {
            box = cv::boundingRect(contours[largestContourIdx]);
        }
    }

    showDebugBoundingBox(scaled_gray, box, "Bounding Box");
    return box;
}

cv::Mat QuickDrawRecognizer::cropImage(const cv::Mat &gray, const cv::Rect &box) {
    int x1 = std::max(0, box.x);
    int y1 = std::max(0, box.y);
    int x2 = std::min(gray.cols, box.x + box.width);
    int y2 = std::min(gray.rows, box.y + box.height);

    cv::Mat cropped = gray(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    showDebugWindow("6. Cropped Image", cropped);

    return cropped;
}

cv::Mat QuickDrawRecognizer::finalizeImage(const cv::Mat &img, int thicken, double pad) {
    cv::Mat result = img.clone();

    // Apply Otsu threshold
    cv::threshold(result, result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    showDebugWindow("7. Otsu Threshold", result);

    // Letterbox to square with padding
    result = letterboxSquare(result, 255, pad);
    showDebugWindow("8. Letterbox Square with Padding", result);

    // Thicken if requested
    if (thicken > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
        cv::erode(result, result, kernel, cv::Point(-1, -1), thicken);
        showDebugWindow("9. Thickened Strokes", result);
    }

    // Resize to 28x28
    cv::resize(result, result, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    showDebugWindow("10. Resized to 28x28", result);

    // Invert (model expects white strokes on black) and normalize to [-1, 1]
    cv::bitwise_not(result, result);
    cv::Mat f32;
    result.convertTo(f32, CV_32F, 1.0f / 255.0f);
    f32 = f32 * 2.0f - 1.0f;

    showDebugWindow("11. Final Normalized", result);

    return f32;
}

void QuickDrawRecognizer::applySoftmax(std::vector<float> &logits) {
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sumExp = 0.f;
    for (float &v : logits) {
        v = std::exp(v - maxLogit);
        sumExp += v;
    }
    for (float &v : logits)
        v /= sumExp;
}

// Debug helper functions
void QuickDrawRecognizer::showDebugWindow(const std::string &windowName, const cv::Mat &image) {
    if (r_config.get<bool>("debug", false)) {
        cv::imshow(windowName, image);
        cv::waitKey(1);
    }
}

void QuickDrawRecognizer::showDebugBoundingBox(const cv::Mat &image, const cv::Rect &box,
                                               const std::string &step) {
    if (r_config.get<bool>("recognizer.debug", false)) {
        cv::Mat debugImage;
        cv::cvtColor(image, debugImage, cv::COLOR_GRAY2BGR);
        cv::rectangle(debugImage, box, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Debug: " + step, debugImage);
        cv::waitKey(1);
    }
}

} // namespace conan
