#include "segment.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {

    LmStatus Segments::addLines(const KeyLines& lines) {
        // Create a segment from each line and compare with existing segments to see if they match. Join if they do, else create new segment
        for (auto lineItr = lines.cbegin(); lineItr != lines.cend; lineItr++) {
            shared_ptr<Segment> lineSegment(new Segment(*lineItr));
            bool joinedToExistingSegment = false;

            for (auto segmentItr = data_.begin(); segmentItr != data_.end(); segmentItr++) {
                shared_ptr<Segment>& pSegment = *segmentItr;
                if (lineSegment->join(*pSegment) == LM_STATUS_OK) {
                    // Successful join operation; update pSegment to the new segment and continue because it is possible for a single line to connect to two segments
                    pSegment = lineSegment;
                    joinedToExistingSegment = true;
                }
            }

            if (!joinedToExistingSegment) {
                data_.push_back(lineSegment);
            }
        }

        return LM_STATUS_OK;
    }

    LmStatus Segments::clear() {
        data_.clear();
        return LM_STATUS_OK;
    }

    Segment::Segment(const cv::line_descriptor::KeyLine& line) {
        data_.push_back(line);
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

    LmStatus Segment::join(Segment& other) {
        SegmentJoint jointType = isJoinedTo(other);

        if (jointType == SEGMENT_JOINT_NONE) {
            return LM_STATUS_ERROR_LINES_UNCONNECTED;
        }

        // We want to append the start of the other segment to the end of this one. We force this by reversing the lists.
        if (!(jointType & 1 << SEGMENT_JOINT_1)) {
            // Connection is made to start of this segment
            data_.reverse();
        }
        if (jointType & 1 << SEGMENT_JOINT_2) {
            // Conection is made to the end of the other segment
            other.data_.reverse();
        }

        // other.data_ is moved to this->data_ and other.data_ is emptied
        data_.splice(data_.end(), other.data_);
        return LM_STATUS_OK;
    }

    SegmentJoint Segment::isJoinedTo(const Segment& other) const {
        bool reverseSelf = false;
        bool reverseOther = false;

        const float connectionDistThresh = 5;

        const KeyLine thisSegmentEnds[2] = {data_.front(), data_.back()};
        const KeyLine otherSegmentEnds[2] = {other.data_.front(), (other.data_.back())};

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (::lm::isJoinedTo(thisSegmentEnds[i], otherSegmentEnds[j], connectionDistThresh)) {
                    return static_cast<SegmentJoint>(i << SEGMENT_JOINT_1 | j << SEGMENT_JOINT_2);
                }
            }
        }    

        return SEGMENT_JOINT_NONE;
    }
}