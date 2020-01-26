#pragma once

// OpenCV2 dependencies
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// Stdlib dependencies

#include "location_matcher/core.hpp"

// Using LSD lines seems to be more stable but parameters need to be tuned for it since it does not detecte some instances of lines (eg a single pixel wide horizontal line). If LSD lines is not used, EDLines is used instead. Currently has memory corruption issues for some reason.
// Current implementation is designed for EDLines. The angle definition of KeyLine needs to be changed for it to work with LSD lines, since angle for LSD line is perpendicular to the line, and angle for EDLines is parallel to the line.
#define USE_LSD_DETECTOR // Comment this out to use EDLines

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

        LmStatus mergeDuplicates(KeyLines lines);

#ifdef USE_LSD_DETECTOR
        cv::Ptr<cv::line_descriptor::LSDDetector> lineDetector_;
#else
        cv::line_descriptor::BinaryDescriptor::Params bdParams_;
        cv::Ptr<cv::line_descriptor::BinaryDescriptor> lineDetector_;
        static int numInstances;        
#endif
    };

}; // namespace lm