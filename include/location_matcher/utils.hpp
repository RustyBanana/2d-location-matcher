#pragma once
#include <opencv2/line_descriptor.hpp>

#include "location_matcher/core.hpp"


namespace lm {

    cv::line_descriptor::KeyLine getKeyLine(float startX, float startY, float endX, float endY);

    cv::Point2f rotateVector(cv::Point2f vec, float angle);

    LineJoint isJoinedTo(const cv::line_descriptor::KeyLine& line1,
                        const cv::line_descriptor::KeyLine& line2,
                        float distThreshold);
}