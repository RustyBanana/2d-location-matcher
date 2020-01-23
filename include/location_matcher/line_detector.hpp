#pragma once

// OpenCV2 dependencies
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <line_descriptor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// Stdlib dependencies

#include "location_matcher/core.hpp"

//#define USE_LSD_DETECTOR

namespace lm {

    class LineDetectorTest;

    class LineDetector {
        friend class LineDetectorTest;
        public:
        LineDetector();
        ~LineDetector();   

        LmStatus detect(const cv::Mat& imgIn, KeyLinesOut lines);

        // Use a mask with the same shape as imgIn, with 1's for pixels to be kept, 0's for pixels to be removed.
        LmStatus detect(const cv::Mat& imgIn, KeyLinesOut lines, const cv::Mat& mask);

        private:
#ifdef USE_LSD_DETECTOR
        cv::Ptr<cv::line_descriptor::LSDDetector> lineDetector_;
#else
        cv::line_descriptor::BinaryDescriptor::Params bdParams_;
        cv::Ptr<cv::line_descriptor::BinaryDescriptor> lineDetector_;        
#endif
    };

}; // namespace lm