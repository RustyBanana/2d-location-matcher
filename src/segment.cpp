#include "location_matcher/segment.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {

    float compareLines(const cv::line_descriptor::KeyLine& line1, const cv::line_descriptor::KeyLine& line2, bool angleInvariant) {
        const float lengthThreshold = 11;
        const float angleThreshold = M_PI * 10/180;

        float angleWeight = angleInvariant ? 1.0f : max(0.0f, 1 - abs(angleDiff(line1.angle, line2.angle))/angleThreshold);

        float lengthWeight = max(0.0f, 1 - abs(line1.lineLength - line2.lineLength)/lengthThreshold);

        return angleWeight * lengthWeight;
    }

    // ######### SegmentMatch ############
    float SegmentMatch::angleThreshold = M_PI * 5/180;
    float SegmentMatch::positionThreshold = 5;

    SegmentMatch::SegmentMatch() {
        
    }

    SegmentMatch::SegmentMatch(Segment seg1, int startIndex1, int endIndex1, Segment seg2, int startIndex2, int endIndex2) {
        
        segment1 = Segment(seg1, startIndex1, endIndex1);
        segment1Index[0] = min(startIndex1, endIndex1);
        segment1Index[1] = max(startIndex1, endIndex1);
        segment2 = Segment(seg2, startIndex2, endIndex2);
        segment2Index[0] = min(startIndex2, endIndex2);
        segment2Index[1] = max(startIndex2, endIndex2);
    }

    LmStatus SegmentMatch::computeOffsets() {
        if (segment1.data_.size() != segment2.data_.size()) {
            return LM_STATUS_SIZE_MISMATCH;
        }

        LmStatus status = LM_STATUS_OK;
        int segmentSize = segment1.data_.size();

        // All angle calculations are done in the range [0, 180] because a line of angle -30deg is a duplicate of an angle of 150deg

        // Get the mean
        float totalAngleMean = 0;
        auto pLine1 = segment1.data_.cbegin();
        auto pLine2 = segment2.data_.cbegin();
        while (pLine1 != segment1.data_.cend()) {
            float angle1 = wrappi(pLine1->angle);
            float angle2 = wrappi(pLine2->angle);

            totalAngleMean += angleDiff(angle2, angle1, M_PI);

            pLine1++;
            pLine2++;
        }
        angleOffset = totalAngleMean/segmentSize;

        // Get the variance
        float totalAngleVariance = 0;
        pLine1 = segment1.data_.cbegin();
        pLine2 = segment2.data_.cbegin();
        while (pLine1 != segment1.data_.cend()) {
            float x, y;
            x = wrappi(pLine1->angle);
            y = wrappi(pLine2->angle);
            x = angleDiff(y-x, angleOffset, M_PI);
            totalAngleVariance += x*x;

            pLine1++;
            pLine2++;
        }

        float angleVariance = totalAngleVariance/(segmentSize-1);
        float angleStdDev = sqrt(angleVariance);
        
        // TODO Validity check based on the std deviation of the angles

        // Get the position offset (naive aprproach)
        // 1. Normalize the position vector of each line by doing it WRT the first line in the segment. This makes it translation invariant.
        vector<Point2f> displacements;
        vector<Point2f> mirroredDisplacements;

        pLine1 = segment1.data_.cbegin();
        pLine2 = segment2.data_.cbegin();
        Point2f line1Pos = pLine1->pt;
        Point2f line2Pos = pLine2->pt;
        pLine1++;
        pLine2++;
        while (pLine1 != segment1.data_.cend()) {
            Point2f d1 = (pLine1->pt - line1Pos);
            Point2f d2 = (pLine2->pt - line2Pos);

            // 2. Use the angle offset to rotate each position vector so that their rotations should match. This makes it rotation invariant.
            d1 = Point2f(  d1.x*cos(angleOffset) + d1.y*-sin(angleOffset),
                            d1.x*sin(angleOffset) + d1.y*cos(angleOffset));
            displacements.push_back(d2 - d1);

            // Get mirrored displacements by mirroring one segment, ie switching y = x and rotating 180deg
            d2 = Point2f(-d2.y, -d2.x);
            mirroredDisplacements.push_back(d2 - d1);

            pLine1++;
            pLine2++;
        }

        // 3. Take an average of the displacement vector from each line in segment1 to the corresponding line in segment2
        Point2f avgDisplacement = accumulate(displacements.begin(), displacements.end(), Point2f(0, 0))/(segmentSize-1);
        Point2f avgMirroredDisplacement = accumulate(mirroredDisplacements.begin(), mirroredDisplacements.end(), Point2f(0, 0))/(segmentSize-1);

        // 4. TODO: Implement validity check based on std dev of displacements
        // Calculate variance
        float totalPositionVariance = accumulate(displacements.begin(), displacements.end(), 0.0f, [](float accumulator, Point2f pt) {
            return accumulator + pt.x*pt.x + pt.y*pt.y;
        });
        float totalMirroredPositionVariance = accumulate(mirroredDisplacements.begin(), mirroredDisplacements.end(), 0.0f, [](float accumulator, Point2f pt) {
            return accumulator + pt.x*pt.x + pt.y*pt.y;
        });

        float positionVariance;
        if (totalPositionVariance < totalMirroredPositionVariance) {
            positionVariance = totalPositionVariance/(segmentSize-1);
            isFlipped = false;
        } else {
            positionVariance = totalMirroredPositionVariance/(segmentSize-1);
            isFlipped = true;
        }
        
        float positionStdDev = sqrt(positionVariance);

        positionOffset = avgDisplacement + line2Pos - line1Pos;

        if (angleStdDev >= angleThreshold || positionStdDev >= positionThreshold) {
            status = LM_STATUS_ERROR_MATCH_FAILED;
        }

        return status;
    }

    // ################### SEGMENT ###################

    Segment::Segment() {
        
    }

    Segment::Segment(const cv::line_descriptor::KeyLine& line) {
        data_.push_back(line);
    }

    Segment::Segment(const Segment& segment, int beginIndex, int endIndex) {
        bool reverse = false;
        if (endIndex < beginIndex) {
            int temp = beginIndex;
            beginIndex = endIndex;
            endIndex = temp;

            reverse = true;
        }

        int i = 0;
        segment_t::const_iterator pData = segment.data_.cbegin();
        while (i < segment.data_.size()) {
            if (i >= beginIndex && i <= endIndex) {
                data_.push_back(*pData);
            }
            pData++;
            i++;
        }
        if (reverse) {
            data_.reverse();
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
    void getMatchIndexes(Iterator thisBegin, Iterator thisEnd, Iterator otherBegin, Iterator otherEnd, Mat likenessMatrix, bool angleInvariant=false) {
        int i = 0;
        int j = 0;
        for (auto thisLineItr = thisBegin; thisLineItr != thisEnd; thisLineItr++) {
            j = 0;
            for (auto otherLineItr = otherBegin; otherLineItr != otherEnd; otherLineItr++) {
                likenessMatrix.at<float>(i, j) = compareLines(*thisLineItr, *otherLineItr, angleInvariant);
                j++;
            }
            i++;
        }
    }

    void Segment::findMatchesInDiag(Mat likenessMatrix, Point2i startIndex, Point2i incrementIndex, float likenessThreshold, const Segment& other, vector<SegmentMatch>& matches) const {
        Point2i prevMatch, currIndex, prevIndex;
        int numRows = likenessMatrix.rows;
        int numCols = likenessMatrix.cols;
        int matchLength = 0;
        int minMatchLength = 2;

        currIndex = startIndex;
        while ( currIndex.y >= 0 && currIndex.y < numCols && 
            currIndex.x >= 0 && currIndex.x < numRows) {
            if (likenessMatrix.at<float>(currIndex) >= likenessThreshold) {
                if (matchLength == 0) {
                    prevMatch = currIndex;
                }
                matchLength++;
            } else {
                if (matchLength >= minMatchLength) {
                    // Matches for current segment ended. Create a segment from prevMatch to prevIndex
                    SegmentMatch newMatch(*this, prevMatch.x, prevIndex.x, other, prevMatch.y, prevIndex.y);
                    if (newMatch.computeOffsets() == LM_STATUS_OK) {
                        matches.push_back(newMatch);
                    }
                }
                matchLength = 0;
            }
            prevIndex = currIndex;
            currIndex += incrementIndex;
        }

        if (matchLength >= minMatchLength) {
            // Matches for current segment ended. Create a segment from prevMatch to prevIndex
            SegmentMatch newMatch(*this, prevMatch.x, prevIndex.x, other, prevMatch.y, prevIndex.y);
            if (newMatch.computeOffsets() == LM_STATUS_OK) {
                matches.push_back(newMatch);
            }
        }
    }

    LmStatus Segment::compareWith(const Segment& other, std::vector<SegmentMatch>& matches) const {
        // First compare each line in this with each line in other to find starting point matches.
        // Contains a float from 0->1 which represents the likeness between the line i of this segment at row i of the matrix, to the line j of the other segment at col j of the matrix

        Mat likenessMatrix = Mat(data_.size(), other.data_.size(), CV_32FC1);;
        getMatchIndexes(data_.cbegin(), data_.cend(), other.data_.cbegin(), other.data_.cend(), likenessMatrix, true);

        cout << "Likeness Matrix" << endl;
        cout << likenessMatrix << endl;

        int numRows = likenessMatrix.rows;
        int numCols = likenessMatrix.cols;

        // Search through diagonials of the matrix for adjacent groups
        // Top left to bottom right diagonals
        for (int i = 0; i < numCols; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(0, i), Point2i(1, 1), 0.4, other, matches);
        }

        // Catch edge case of a 1x1 matrix
        if (numRows == 1 && numCols == 1) {
            return LM_STATUS_OK;
        }

        for (int i = 1; i < numRows; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(i, 0), Point2i(1, 1), 0.4, other, matches);
        }

        // Top right to bottom left diagonals
        for (int i = 0; i < numCols; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(0, i), Point2i(1, -1), 0.4, other, matches);
        }
        for (int i = 1; i < numRows; i++) {
            findMatchesInDiag(likenessMatrix, Point2i(i, numCols-1), Point2i(1, -1), 0.4, other, matches);
        }

        return LM_STATUS_OK;

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

    void Segment::draw(InputOutputArray imgIn, Scalar color, string label) const {
        for (auto lineItr = data_.cbegin(); lineItr != data_.cend(); lineItr++) {
            const KeyLine& line = *lineItr;

            cv::line(imgIn, line.getStartPoint(), line.getEndPoint(), color);
            putText(imgIn, label, line.pt, FONT_HERSHEY_SIMPLEX , 1.0, color);
        
        }
    }

    // Getters and setters
    const Segment::segment_t& Segment::data() const {
        return data_;
    }

    // ############ SEGMENTS ############

    LmStatus Segments::addSegment(const Segment& segment) {
        shared_ptr<Segment> newSegment(new Segment(segment));
        data_.push_back(newSegment);
        return LM_STATUS_OK;
    }

    LmStatus Segments::addLines(const KeyLines& lines) {
        // Create a segment from each line and compare with existing segments to see if they match. Join if they do, else create new segment
        
        // True if the segment is unique, false if segment has been merged elsewhere
        vector<bool> isUnique(data_.size(), true);

        for (auto lineItr = lines.cbegin(); lineItr != lines.cend(); lineItr++) {
            shared_ptr<Segment> lineSegment(new Segment(*lineItr));
            bool joinedToExistingSegment = false;

            int segmentCount = 0;
            for (auto segmentItr = data_.begin(); segmentItr != data_.end(); segmentItr++) {
                if (isUnique[segmentCount]) {
                    shared_ptr<Segment>& pSegment = *segmentItr;
                    if (lineSegment->join(*pSegment) == LM_STATUS_OK) {
                        // Successful join operation; update pSegment to the new segment and continue because it is possible for a single line to connect to two segments
                        pSegment = lineSegment;
                        joinedToExistingSegment = true;
                        isUnique[segmentCount] = false;
                    }
                }
                segmentCount++;
            }

            data_.push_back(lineSegment);
            isUnique.push_back(true);
        }
        pruneSegments(isUnique);

        return LM_STATUS_OK;
    }

    LmStatus Segments::clear() {
        data_.clear();
        return LM_STATUS_OK;
    }

    LmStatus Segments::matchSegments(const Segments& segments, std::vector<SegmentMatch>& matches) {
        for (auto otherSegmentItr = segments.data().cbegin(); otherSegmentItr != segments.data().cend(); otherSegmentItr++) {
            Segment& otherSegment = **otherSegmentItr;

            for (auto thisSegmentItr = data_.cbegin(); thisSegmentItr != data_.cend(); thisSegmentItr++) {
                Segment& thisSegment = **thisSegmentItr;

                thisSegment.compareWith(otherSegment, matches);
            }
        }

        return LM_STATUS_OK;
    }

    void Segments::draw(cv::InputOutputArray imgIn) const {
        srand(time(NULL));

        int count = 0;
        for (auto segmentItr = data_.cbegin(); segmentItr != data_.cend(); segmentItr++) {
            Segment& segment = **segmentItr;

            int r = rand() % 255 + 1;
            int g = rand() % 255 + 1;
            int b = rand() % 255 + 1;

            Scalar color(b,g,r);
            string label = to_string(count);

            segment.draw(imgIn, color, label);
            
            count++;
        }
    }

    const Segments::data_t& Segments::data() const {
        return data_;
    }

    void Segments::pruneSegments(std::vector<bool> keepList) {
        std::vector<std::shared_ptr<Segment>> prunedSegments;
        auto pSegmentItr = data_.cbegin();
        auto keepItr = keepList.cbegin();

        while (pSegmentItr != data_.cend()) {
            if (*keepItr == true) {
                prunedSegments.push_back(*pSegmentItr);
            }
            pSegmentItr++;
            keepItr++;
        }
        data_ = prunedSegments;
    }
}