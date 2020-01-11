#pragma once

#include "opencv2/core.hpp"

#include "location_matcher/core.hpp"

namespace lm {
    class LineFilterTest;

    class LineFilter {
        friend class LineFilterTest;
        public:
        LineFilter(double minLineLength, double maxLineLength);
        ~LineFilter();

        LmStatus filterByLine(Mat& imgIn, Mat& imgOut, KeyLinesIn lines,  const int lineThickness);

        void setMinLineLength(double minLength);
        void setMaxLineLength(double maxLength);

        private:
        double minLineLength_;
        double maxLineLength_;
    };
}; // namespace lm