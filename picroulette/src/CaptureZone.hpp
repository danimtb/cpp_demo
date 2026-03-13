#pragma once

#include "Config.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace conan {

class CaptureZone {
public:
    explicit CaptureZone(const Config &cfg, int winW, int winH) {
        r_origX = cfg.get<int>("captureZone.x", 50);
        r_origY = cfg.get<int>("captureZone.y", 100);
        r_origW = cfg.get<int>("captureZone.width", 400);
        r_origH = cfg.get<int>("captureZone.height", 300);
        r_normX = float(r_origX) / winW;
        r_normY = float(r_origY) / winH;

        r_minContentArea = cfg.get<int>("captureZone.minContentArea", 1000);
        r_stabilityFrames = cfg.get<int>("captureZone.stabilityFrames", 5);
        r_stabilityProgress = 0;
        r_zoneActive = false;
        r_recognitionThreshold = cfg.get<float>("captureZone.recognitionThreshold", 0.5f);

        // Fixed rectangle in pixels
        r_zone = cv::Rect(r_origX, r_origY, r_origW, r_origH);
    }

    void updateWindowSize(int winW, int winH) {
        // Update rectangle position relative to new window size
        r_zone.x = int(r_normX * winW);
        r_zone.y = int(r_normY * winH);
        // r_zone.width / height remain fixed
    }

    const cv::Rect &getZone() const { return r_zone; }

    bool hasContentInZone(const cv::Mat &frame, int winW, int winH) const {
        cv::Rect roiRect = mapToFrame(frame, winW, winH);
        if (roiRect.width <= 0 || roiRect.height <= 0) return false;

        cv::Mat roi = frame(roiRect);
        cv::Mat gray;
        if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        else gray = roi;

        cv::Mat thresh;
        cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        int contentArea = cv::countNonZero(thresh);
        return contentArea >= r_minContentArea;
    }

    void updateZoneState(const cv::Mat &frame, int winW, int winH) {
        bool hasContent = hasContentInZone(frame, winW, winH);
        if (hasContent) {
            r_stabilityProgress++;
            if (r_stabilityProgress >= r_stabilityFrames) r_zoneActive = true;
        } else {
            r_stabilityProgress = 0;
            r_zoneActive = false;
        }
    }

    cv::Rect mapToFrame(const cv::Mat &frame, int winW, int winH) const {
        float scale = std::min(float(winW) / frame.cols, float(winH) / frame.rows);
        int scaledW = int(frame.cols * scale);
        int scaledH = int(frame.rows * scale);
        int offsetX = (winW - scaledW) / 2;
        int offsetY = (winH - scaledH) / 2;

        int x = int((r_zone.x - offsetX) / scale);
        int y = int((r_zone.y - offsetY) / scale);
        int w = int(r_zone.width / scale);
        int h = int(r_zone.height / scale);

        x = frame.cols - x - w; // horizontal mirror

        return cv::Rect(x, y, w, h) & cv::Rect(0, 0, frame.cols, frame.rows);
    }

    bool isActive() const { return r_zoneActive; }
    int getStabilityProgress() const { return r_stabilityProgress; }
    float getRecognitionThreshold() const { return r_recognitionThreshold; }

private:
    int r_origX, r_origY, r_origW, r_origH;
    float r_normX, r_normY;
    cv::Rect r_zone;

    int r_minContentArea;
    int r_stabilityFrames;
    int r_stabilityProgress;
    bool r_zoneActive;
    float r_recognitionThreshold;
};

} // namespace conan
