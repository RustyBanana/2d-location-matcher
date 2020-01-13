#pragma once

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>

namespace lm {
    enum LmStatus {
        LM_STATUS_OK = 0,
        LM_STATUS_ERROR_GENERIC,
        LM_STATUS_ERROR_LINES_UNCONNECTED,
        LM_STATUS_SIZE_MISMATCH
    };

    inline float dist(cv::Point2f pt1, cv::Point2f pt2) {
        cv::Point2f diff = pt2 - pt1;
        return sqrt(diff.dot(diff));
    }

    inline float wrap2pi(float angle) {
        return fmod(angle, M_PI);
    }

    inline float angleDiff(float angle1, float angle2) {
        float diff = angle1 - angle2;
        if (abs(diff <= M_PI)) {
            return diff;
        } else {
            return  diff < 0 ? diff + 2 * M_PI : diff - 2 * M_PI;
        }
    }

    typedef std::vector<cv::line_descriptor::KeyLine> KeyLines;

    typedef const KeyLines& KeyLinesIn;
    typedef KeyLines& KeyLinesOut;

    // Each Keyline in the list is placed adjacent to the KeyLines they are connected to in the image.
    typedef std::list<std::shared_ptr<cv::line_descriptor::KeyLine>> Segment11;


};