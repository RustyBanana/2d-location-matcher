#include "location_matcher/segment.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {
    
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

    float compareLines(const cv::line_descriptor::KeyLine& line1, const cv::line_descriptor::KeyLine& line2) {
        const float lengthThreshold = 10;
        const float angleThreshold = M_PI * 10/180;

        float angleWeight = max(0.0f, abs(angleDiff(line1.angle, line2.angle))/angleThreshold);

        float lengthWeight = max(0.0f, abs(line1.lineLength - line2.lineLength)/lengthThreshold);

        return angleWeight * lengthWeight;
    }

    LmStatus SegmentMatch::computeOffsets() {
        if (segment1.data_.size() != segment2.data_.size()) {
            return LM_STATUS_SIZE_MISMATCH;
        }

        float angle1 = 0;
        float angle2 = 0;

        // Do an averaging operation for the angle of each point
        auto pLine1 = segment1.data_.cbegin();
        auto pLine2 = segment2.data_.cbegin();
        while (pLine1 != segment1.data_.cend()) {
            angle1 += pLine1->angle;
            angle2 += pLine2->angle;

            pLine1++;
            pLine2++;
        }
        angle1 /= segment1.data_.size();
        angle2 /= segment2.data_.size();

        // TODO implement angle offset and position offset

    }

    LmStatus Segments::addLines(const KeyLines& lines) {
        // Create a segment from each line and compare with existing segments to see if they match. Join if they do, else create new segment
        for (auto lineItr = lines.cbegin(); lineItr != lines.cend(); lineItr++) {
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

    Segment::Segment() {
        
    }

    Segment::Segment(const cv::line_descriptor::KeyLine& line) {
        data_.push_back(line);
    }

    Segment::Segment(const Segment& segment, int beginIndex, int endIndex) {
        int i = 0;
        auto pData = segment.data_.begin();
        while (i < segment.data_.size()) {
            if (i >= beginIndex && i < endIndex) {
                data_.push_back(*pData);
            }
            pData++;
            i++;
        }
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
                    return static_cast<SegmentJoint>(
                            i << SEGMENT_JOINT_1 |
                            j << SEGMENT_JOINT_2 |
                            1 << SEGMENT_JOINT_JOINED);
                }
            }
        }    

        return SEGMENT_JOINT_NONE;
    }

    template <typename Iterator>
    void getMatchIndexes(Iterator thisBegin, Iterator thisEnd, Iterator otherBegin, Iterator otherEnd, vector<vector<int>>& matchIndexes, Mat likenessMatrix) {
        const float likenessThreshold = 0.4;

        int i = 0;
        int j = 0;
        for (auto thisLineItr = thisBegin; thisLineItr != thisEnd; thisLineItr++) {
            j = 0;
            for (auto otherLineItr = otherBegin; otherLineItr != otherEnd; otherLineItr++) {
                likenessMatrix.at<float>(i, j) = compareLines(*thisLineItr, *otherLineItr);
                if (compareLines(*thisLineItr, *otherLineItr) >= likenessThreshold) {
                    matchIndexes.push_back(vector<int>(i, j));
                }
                j++;
            }
            i++;
        }
    }

    void Segment::findMatchesInDiag(Mat likenessMatrix, Point2i startIndex, Point2i incrementIndex, float likenessThreshold, vector<SegmentMatch>& matches) const {
        Point2i prevMatch, currIndex, prevIndex;
        int numRows = likenessMatrix.rows;
        int numCols = likenessMatrix.cols;
        int matchLength = 0;
        currIndex = startIndex;
        while ( currIndex.y >= 0 && currIndex.y < numCols && 
            currIndex.x >= 0 && currIndex.x < numRows) {
            if (likenessMatrix.at<float>(currIndex) >= likenessThreshold) {
                if (matchLength == 0) {
                    prevMatch = currIndex;
                }
                matchLength++;
            } else {
                if (matchLength > 1) {
                    // Matches for current segment ended. Create a segment fro prevMatch to prevIndex
                    SegmentMatch newMatch;
                    newMatch.segment1 = Segment(*this, prevMatch.x, prevIndex.x);
                    newMatch.segment2 = Segment(*this, prevMatch.y, prevIndex.y);
                    newMatch.computeOffsets();
                    matches.push_back(newMatch);
                }
                matchLength = 0;
            }
            currIndex += incrementIndex;
        }
    }

    LmStatus Segment::compareWith(const Segment& other, std::vector<SegmentMatch>& matches) const {
        // First compare each line in this with each line in other to find starting point matches.
        Mat likenessMatrix = Mat(data_.size(), other.data_.size(), CV_32FC1);;
        vector<vector<int>> matchIndexes;
        getMatchIndexes(data_.cbegin(), data_.cend(), other.data_.cbegin(), other.data_.cend(), matchIndexes, likenessMatrix);

        int numRows = likenessMatrix.rows;
        int numCols = likenessMatrix.cols;

        // Search through diagonials of the matrix for adjacent groups
        // Top left to bottom right diagonals
        for (int i = 0; i < numCols; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(0, i), Point2i(1, 1), 0.4, matches);
        }
        for (int i = 1; i < numRows; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(i, 0), Point2i(1, 1), 0.4, matches);
        }

        // Top right to bottom left diagonals
        for (int i = 0; i < numCols; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(i, 0), Point2i(-1, 1), 0.4, matches);
        }
        for (int i = 1; i < numRows; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(numCols-1, i), Point2i(-1, 1), 0.4, matches);
        }

        /*
        // If there are strong diagonals in the matrix, then there are matches there. We check for this by adding and subtracting the row and column indexes stored in matchIndexes and checking for any repeated constant values. diagonals going from top left to bottom right give a constant for row - column, diagonals from bottom left to top right give a constant for row + column.
        // Create a vector of shape (2,0). Store sums at [0] and diffs at [1]
        vector<vector<int>> indexSums({{},{}});
        for (auto itr = matchIndexes.cbegin(); itr != matchIndexes.cend(); itr++) {
            indexSums[0].push_back((*itr)[0] + (*itr)[1]);
            indexSums[1].push_back((*itr)[0] - (*itr)[1]);
        }

        // Sort each vector look for duplicates.
        vector<vector<int>> duplicates({{},{}});
        for (int i = 0; i < 2; i++) {
            sort(indexSums[i].begin(), indexSums[i].end());
            int prevNum = -99999;
            int prevDuplicate = -99999;
            for (auto itr = indexSums[i].cbegin(); itr != indexSums[i].cend(); itr++) {
                int currNum = *itr;
                if (prevNum == currNum && currNum != prevDuplicate) {
                    duplicates[i].push_back(currNum);
                    prevDuplicate = currNum;
                }
                prevNum = currNum;
            }
        }
        
        // For each duplicate, search through the matchIndexes vector to looking for values which are adjacent. If they are found, create a new segment match for it.
        for (auto itr = duplicates[0].cbegin(); itr != duplicates[0].cend(); itr++) {
            int sum = *itr;

        }
        */
    }

    // Getters and setters
    const Segment::segment_t& Segment::data() {
        return data_;
    }
}