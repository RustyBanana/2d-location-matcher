#include "location_matcher/line_filter.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {

    LineFilter::LineFilter(double minLineLength, double maxLineLength) : minLineLength_(minLineLength), maxLineLength_(maxLineLength) {

    }

    LmStatus LineFilter::filterByLine(const Mat& imgIn, Mat& imgOut, KeyLinesIn lines,  const int lineThickness) {
        Mat mask = Mat::zeros(imgIn.size(), CV_8UC1);

        // Paint every line in lines on to the mask as 0xff to prepare for bitwise_and
        for (auto lineItr = lines.cbegin(); lineItr != lines.cend(); lineItr++) {
            
            KeyLine kl = *lineItr;
        
            Point2f pt1 = Point2f( kl.startPointX, kl.startPointY );
            Point2f pt2 = Point2f( kl.endPointX, kl.endPointY );

            Point2f lineVec = pt2 - pt1;
            float length = sqrt(abs(lineVec.dot(lineVec)));

            if (length >= minLineLength_ && length <= maxLineLength_) {
                line( mask, pt1, pt2, Scalar( 255 ), lineThickness );
            }
        }

        imgOut = Mat(imgIn.size(), CV_8UC1, Scalar(255));
        imgIn.copyTo(imgOut, mask);

        return LM_STATUS_OK;
    }

    void LineFilter::setMinLineLength(double minLength) {
        minLineLength_ = minLength;
    }
    void LineFilter::setMaxLineLength(double maxLength) {
        maxLineLength_ = maxLength;
    }

};  // namespace lm