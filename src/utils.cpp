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
}