#include "location_matcher/utils.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm{
    KeyLine getKeyLine(float startX, float startY, float endX, float endY) {
        KeyLine kl;

        kl.startPointX  = startX;
        kl.startPointY  = startY;
        kl.endPointX    = endX;
        kl.endPointY    = endY;
        Point2f klVec = kl.getEndPoint() - kl.getStartPoint();
        kl.angle        = atan2(klVec.y, klVec.x);
        kl.lineLength   = sqrt(klVec.dot(klVec));
        kl.pt = (kl.getStartPoint() + kl.getEndPoint())/2;

        return kl;
    }

    cv::Point2f rotateVector(cv::Point2f vec, float angle) {
        return Point2f( vec.x*cos(angle) - vec.y*sin(angle),
                        vec.x*sin(angle) + vec.y*cos(angle));
    }

    LineJoint isJoinedTo(const cv::line_descriptor::KeyLine& line1,
                         const cv::line_descriptor::KeyLine& line2,
                         float distThreshold) {
        Point2f start1(line1.startPointX, line1.startPointY);
        Point2f start2(line2.startPointX, line2.startPointY);
        Point2f end1(line1.endPointX, line1.endPointY);
        Point2f end2(line2.endPointX, line2.endPointY);
        
        return dist(start1, start2) <= distThreshold ? LINE_JOINT_SS :
               dist(start1, end2)   <= distThreshold ? LINE_JOINT_SE :
               dist(end1,   start2) <= distThreshold ? LINE_JOINT_ES :
               dist(end1,   end2)   <= distThreshold ? LINE_JOINT_EE :
               LINE_JOINT_NONE;
    }

    void drawLines(Mat& output, KeyLines lines) {
        if( output.channels() == 1 )
            cvtColor( output, output, COLOR_GRAY2BGR );
        for ( size_t i = 0; i < lines.size(); i++ )
        {
            KeyLine kl = lines[i];
            if( kl.octave == 0)
            {
                /* get a random color */
                int R = ( rand() % (int) ( 255 + 1 ) );
                int G = ( rand() % (int) ( 255 + 1 ) );
                int B = ( rand() % (int) ( 255 + 1 ) );

                /* get extremes of line */
                Point pt1 = Point2f( kl.startPointX, kl.startPointY );
                Point pt2 = Point2f( kl.endPointX, kl.endPointY );

                /* draw line */
                line( output, pt1, pt2, Scalar( B, G, R ), 3 );

                std::ostringstream oss;
                oss << i;
                putText(output, oss.str(), kl.pt, cv::HersheyFonts::FONT_HERSHEY_PLAIN, 0.8, Scalar(B, G, R));
            }
        }
    }
}