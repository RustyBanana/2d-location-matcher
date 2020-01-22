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
}