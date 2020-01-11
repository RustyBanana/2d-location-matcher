#pragma once
#include <opencv2/line_descriptor.hpp>

#include "location_matcher/core.hpp"


namespace lm {
    cv::line_descriptor::KeyLine getKeyLine(float startX, float startY, float endX, float endY);
}