#include "location_matcher/line_detector.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {  
    LineDetector::LineDetector() {
        BinaryDescriptor::Params bdParams;
        bdParams.ksize_ = 5;            // Gaussian kernel size. Higher = less 
                                        // sensitivity of disjointed sections
        bdParams.numOfOctave_ = 1;      // Number of octaves in image pyramid
        bdParams.reductionRatio = 2;    // Image reduction ratio when 
                                        // constructing gaussian pyramid octaves
        bdParams.widthOfBand_ = 1;

#ifdef USE_LSD_DETECTOR
        lineDetector_ = LSDDetector::createLSDDetector();
#else
        bdParams_ = bdParams;
        lineDetector_ = BinaryDescriptor::createBinaryDescriptor(bdParams);
#endif
    }      
    
    LmStatus LineDetector::detect(const cv::Mat& imgIn, KeyLinesOut lines) {
        cv::Mat mask = cv::Mat::ones(imgIn.size(), CV_8UC1);
        return detect(imgIn, lines, mask);
    }

    LmStatus LineDetector::detect(const cv::Mat& imgIn, KeyLinesOut lines, const cv::Mat& mask) {
        KeyLines detectedLines;
#ifdef USE_LSD_DETECTOR
        lineDetector_->detect(imgIn, detectedLines,  1, 1, mask);
#else
        lineDetector_->detect(imgIn, detectedLines, mask );
#endif
        float offsetR = 1.6; // Estimation of impact of gaussian kernel smoothing on line position
        for ( size_t i = 0; i < detectedLines.size(); i++ )
        {
            KeyLine kl = detectedLines[i];
            
            if( kl.octave == 0 && kl.angle >= 0)
            {

                /* get extremes of line */
                Point2f pt1 = Point2f( kl.startPointX, kl.startPointY );
                Point2f pt2 = Point2f( kl.endPointX, kl.endPointY );

                // Shift the line by an offset because of the detection position bias of gaussian kernel smoothing
                float angle = kl.angle;
                angle = M_PI - angle;
                Point2f offset = Point2f(offsetR * sin(angle), offsetR * cos(angle));
                pt1 = pt1 + offset;
                pt2 = pt2 + offset;
                kl.startPointX = pt1.x;
                kl.startPointY = pt1.y;
                kl.endPointX = pt2.x;
                kl.endPointY = pt2.y;
                kl.pt = kl.pt + offset;
                lines.push_back(kl);
            }
        }
        return LM_STATUS_OK;
    }
} // namespace bzd