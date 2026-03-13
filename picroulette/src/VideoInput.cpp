#include "VideoInput.h"
#include "Config.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>

namespace conan {
VideoInput::VideoInput(const Config &cfg) {
    const auto videoPath = cfg.get<std::string>("ui.videoPath", "");
    if (!videoPath.empty()) {
        r_isCamera = false;
        r_cap.open(assetPath(videoPath));
    } else {
        r_isCamera = true;
        const int device = cfg.get<int>("ui.cameraIndex", 0);
        r_cap.open(device);
    }
    if (!r_cap.isOpened())
        throw std::runtime_error("Failed to open video source");

    const int desiredW = cfg.get<int>("ui.cameraWidth", 960);
    const int desiredH = cfg.get<int>("ui.cameraHeight", 540);

    r_cap.set(cv::CAP_PROP_FRAME_WIDTH, desiredW);
    r_cap.set(cv::CAP_PROP_FRAME_HEIGHT, desiredH);
}

bool VideoInput::getFrame(cv::Mat &frame) {
    r_cap >> frame;
    if (frame.empty() && !r_isCamera) {
        r_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        r_cap >> frame;
    }
    return !frame.empty();
}

} // namespace conan
