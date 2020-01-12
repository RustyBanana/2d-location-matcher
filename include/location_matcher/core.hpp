#pragma once

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>

namespace lm {
    enum LmStatus {
        LM_STATUS_OK = 0,
        LM_STATUS_ERROR_GENERIC,
        LM_STATUS_ERROR_LINES_UNCONNECTED
    };

    inline float dist(cv::Point2f pt1, cv::Point2f pt2) {
        Point2f diff = pt2 - pt1;
        return sqrt(diff.dot(diff));
    }

    typedef std::vector<cv::line_descriptor::KeyLine> KeyLines;

    typedef const KeyLines& KeyLinesIn;
    typedef KeyLines& KeyLinesOut;

    // Each Keyline in the list is placed adjacent to the KeyLines they are connected to in the image.
    typedef std::list<std::shared_ptr<cv::line_descriptor::KeyLine>> Segment11;
    typedef std::vector<Segment> segments;


};