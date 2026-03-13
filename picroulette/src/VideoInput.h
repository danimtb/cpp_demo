#pragma once

#include "Config.hpp"
#include <opencv2/opencv.hpp>

namespace conan {
class VideoInput {
  public:
    VideoInput(const Config &cfg);
    ~VideoInput() { r_cap.release(); }
    bool getFrame(cv::Mat &frame);
    inline double getFrameWidth() const { return r_cap.get(cv::CAP_PROP_FRAME_WIDTH); }
    inline double getFrameHeight() const { return r_cap.get(cv::CAP_PROP_FRAME_HEIGHT); }

  private:
    cv::VideoCapture r_cap;
    bool r_isCamera;
};
} // namespace conan
