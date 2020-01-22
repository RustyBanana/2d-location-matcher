#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "test/test_datasets.hpp"
#include "location_matcher/utils.hpp"
#include "location_matcher/segment.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {

    void EXPECT_SEGMENT_EQUAL(const Segment& seg1, const Segment& seg2) {
        ASSERT_EQ(true, seg1.data().size() == seg2.data().size());

        auto pSeg1 = seg1.data().cbegin();
        auto pSeg2 = seg2.data().cbegin();

        // Reverse seg2 if the first line in seg1 and seg2 don't match.
        LineJoint lj = isJoinedTo(seg1.data().front(), seg2.data().front(), 6);
        if (lj == LINE_JOINT_NONE) {
            pSeg2 = seg2.data().crbegin().base();
            pSeg2++; // Increment1 becase crbegin() is the same as cend();
        }

        while (pSeg1 != seg1.data().cend()) {
            EXPECT_KEYLINE_EQUAL(*pSeg1, *pSeg2);
            pSeg1++;
            pSeg2++;
        }
    }

    class SegmentTest : public BaseTest {
        public:
        // Used to generate a vector from connected KeyLines
        Segment autogenSegment(KeyLines lines) {
            Segment seg;
            for (auto lineItr = lines.cbegin(); lineItr != lines.cend(); lineItr++) {
                seg.data_.push_back(*lineItr);
            }
            return seg;
        }

        vector<Segment> getSegmentsVec(KeyLines lines) {
            vector<Segment> segVec;
            for (int i = 0; i < lines.size(); i++) {
                segVec.push_back(Segment(lines[i]));
            }

            return segVec;
        }

        void SetUp() override {
            BaseTest::SetUp();
            
            segmentsVec1_ = getSegmentsVec(lines1_);
            segmentsVec2_ = getSegmentsVec(lines2_);
            segmentsVec3_ = getSegmentsVec(lines3_);
            segmentsVec4_ = getSegmentsVec(lines4_);
            segmentsVec5_ = getSegmentsVec(lines5_);

            segmentsVecAns1_ = segmentsVec1_;
            segmentsVecAns2_.push_back(segmentsVec2_[0]);
            segmentsVecAns2_[0].data_.push_back(segmentsVec2_[1].data_.front());
            segmentsVecAns2_.push_back(segmentsVec2_[2].data_.front());

            // These lines are all connected together with the last line being the first in the segment.
            segmentsVecAutogen3_.push_back(autogenSegment(lines3_));
            segmentsVecAutogen4_.push_back(autogenSegment(lines4_));
            segmentsVecAutogen5_.push_back(autogenSegment(lines5_));
        }

        // Vector containing segments with a single line each
        vector<Segment> segmentsVec1_;
        vector<Segment> segmentsVec2_;
        vector<Segment> segmentsVec3_;
        vector<Segment> segmentsVec4_;
        vector<Segment> segmentsVec5_;

        // Manually created resultant segment vector when segmentsVecX_ is joined together
        vector<Segment> segmentsVecAns1_;
        vector<Segment> segmentsVecAns2_;

        // Automtatically generated segment vectors
        vector<Segment> segmentsVecAutogen3_;
        vector<Segment> segmentsVecAutogen4_;
        vector<Segment> segmentsVecAutogen5_;

    };

    class SegmentsTest : public SegmentTest {
        public:
        void SetUp() override {
            SegmentTest::SetUp();

            for (auto segItr = segmentsVecAns1_.cbegin(); segItr != segmentsVecAns1_.cend(); segItr++) {
                segments1_.data_.push_back(shared_ptr<Segment>(new Segment(*segItr)));
            }

            for (auto segItr = segmentsVecAns2_.cbegin(); segItr != segmentsVecAns2_.cend(); segItr++) {
                segments2_.data_.push_back(shared_ptr<Segment>(new Segment(*segItr)));
            }
            
        }

        Segments segments1_;
        Segments segments2_;
    };

    class SegmentMatchTest : public SegmentsTest {
        public:
        void SetUp() override {
            SegmentsTest::SetUp();
        }
    };

    class SegmentIntegrationTest : public BaseTest {
        void SetUp() override {
            BaseTest::SetUp();
        }
        
    };

    TEST_F(SegmentTest, isJoinedToUnconnected) {
        EXPECT_EQ(SEGMENT_JOINT_NONE, segmentsVec1_[0].isJoinedTo(segmentsVec1_[1]));
    }

    TEST_F(SegmentTest, isJoinedToConnected) {
        EXPECT_EQ(SEGMENT_JOINT_FF, segmentsVec2_[0].isJoinedTo(segmentsVec2_[1]));
    }

    TEST_F(SegmentTest, joinUnconnected) {
        vector<Segment> segmentsVec = segmentsVec1_;
        LmStatus status = segmentsVec[0].join(segmentsVec[1]);

        EXPECT_EQ(LM_STATUS_ERROR_LINES_UNCONNECTED, status);
        EXPECT_KEYLINE_EQUAL(segmentsVec[0].data().front(), segmentsVec1_[0].data().front());
        EXPECT_KEYLINE_EQUAL(segmentsVec[1].data().front(), segmentsVec1_[1].data().front());
    }

    TEST_F(SegmentTest, joinConnected) {
        vector<Segment> segmentsVec = segmentsVec2_;

        LmStatus status = segmentsVec[0].join(segmentsVec[1]);
 
        EXPECT_EQ(LM_STATUS_OK, status);
        EXPECT_KEYLINE_EQUAL(segmentsVec[0].data().front(), segmentsVec2_[0].data().front());
        EXPECT_KEYLINE_EQUAL(segmentsVec[0].data().back(), segmentsVec2_[1].data().front());
        EXPECT_EQ(true, segmentsVec[1].data().empty());
    }

    TEST_F(SegmentTest, joinLargeWallInterior2) {
        vector<Segment> segmentsVec = segmentsVec4_;

        for (int i = 1; i < segmentsVec.size(); i++) {
            LmStatus status = segmentsVec[0].join(segmentsVec[i]);
            EXPECT_EQ(LM_STATUS_OK, status);
            EXPECT_EQ(true, segmentsVec[i].data().empty());
        }

        EXPECT_SEGMENT_EQUAL(segmentsVec[0], segmentsVecAutogen4_[0]);
    }

/*  REMOVED UNTIL IMPLEMENTATION OF MIN MATCH LENGTH = 1
    TEST_F(SegmentTest, compareWithUnconnectedMatched) {
        Segment segment1Copy = segmentsVecAns1_[0];
        
        std::vector<SegmentMatch> matches;
        LmStatus status = segment1Copy.compareWith(segmentsVecAns1_[0], matches);

        EXPECT_EQ(LM_STATUS_OK, status);

        // We compare a segment with a single line to itself.
        EXPECT_SEGMENT_EQUAL(segmentsVecAns1_[0], matches[0].segment1);
        EXPECT_SEGMENT_EQUAL(segmentsVecAns1_[0], matches[0].segment2);
    }
*/

    TEST_F(SegmentTest, compareWithUnconnectedUnmatched) {
        Segment randomSegment = Segment(getKeyLine(0, 0, 100, 100));
        
        std::vector<SegmentMatch> matches;
        LmStatus status = randomSegment.compareWith(segmentsVecAns1_[0], matches);

        EXPECT_EQ(LM_STATUS_OK, status);

        // We compare a segment with a single line to a different line of same length
        EXPECT_EQ(0, matches.size());
    }

    TEST_F(SegmentTest, compareWithConnectedMatched) {
        Segment segment2Copy = segmentsVecAns2_[0];
        std::vector<SegmentMatch> matches;

        LmStatus status = segment2Copy.compareWith(segmentsVecAns2_[0], matches);
        EXPECT_EQ(LM_STATUS_OK, status);

        // We compare a segment of 2 joined lines to itself
        EXPECT_SEGMENT_EQUAL(segmentsVecAns2_[0], matches[0].segment1);
        EXPECT_SEGMENT_EQUAL(segmentsVecAns2_[0], matches[0].segment2);

    }
    
    TEST_F(SegmentTest, compareWithConnectedMatchedRotated45) {
        Segment segment2Copy = segmentsVecAns2_[0];

        
        std::vector<SegmentMatch> matches;

        LmStatus status = segment2Copy.compareWith(segmentsVecAns2_[0], matches);
        EXPECT_EQ(LM_STATUS_OK, status);

        // We compare a segment of 2 joined lines to itself
        EXPECT_SEGMENT_EQUAL(segmentsVecAns2_[0], matches[0].segment1);
        EXPECT_SEGMENT_EQUAL(segmentsVecAns2_[0], matches[0].segment2);
    }

    TEST_F(SegmentsTest, addLinesUnconnected) {
        Segments segments;
        LmStatus status = segments.addLines(lines1_);

        EXPECT_EQ(LM_STATUS_OK, status);

        auto pSegment = segments.data()[0];
        EXPECT_KEYLINE_EQUAL(pSegment->data().front(), segmentsVec1_[0].data().front());
        pSegment = segments.data()[1];
        EXPECT_KEYLINE_EQUAL(pSegment->data().front(), segmentsVec1_[1].data().front());
    }

    TEST_F(SegmentsTest, addLinesConnected) {
        Segments segments;
        LmStatus status = segments.addLines(lines2_);

        EXPECT_EQ(LM_STATUS_OK, status);

        auto pSegment = segments.data()[0];
        EXPECT_KEYLINE_EQUAL(pSegment->data().front(), segmentsVec2_[1].data().front());
        pSegment = segments.data()[0];
        EXPECT_KEYLINE_EQUAL(pSegment->data().back(), segmentsVec2_[0].data().front());
        pSegment = segments.data()[1];
        EXPECT_KEYLINE_EQUAL(pSegment->data().front(), segmentsVec2_[2].data().front());        
    }

    // ########### SegmentMatch Test ############
    TEST_F(SegmentMatchTest, computeOffsetsIdentical) {
        SegmentMatch match;
        match.segment1 = segmentsVecAns2_[0];
        match.segment2 = segmentsVecAns2_[0];

        LmStatus status = match.computeOffsets();

        EXPECT_EQ(LM_STATUS_OK, status);
        EXPECT_EQ(0, match.angleOffset);
        EXPECT_EQ(Point2f(0,0), match.positionOffset);
    }

    TEST_F(SegmentMatchTest, computeOffsetsAngleValue) {
        // Test angle offset only
        SegmentMatch match;
        match.segment1 = segmentsVecAutogen4_[0];
        match.segment2 = segmentsVecAutogen5_[0];

        match.computeOffsets();

        EXPECT_NEAR(M_PI_4, match.angleOffset, 1e-5);
        
    }

    TEST_F(SegmentMatchTest, computeOffsetsAngleValidity) {
        // Test angle offset validity only
        EXPECT_STREQ("NOT_IMPLEMENTED", "TODO");
    }

    TEST_F(SegmentMatchTest, computeOffsetsPositionValue) {
        // Test position offset value only
        KeyLines movedLines = lines2_;
        movedLines[0].endPointX += 10;
        movedLines[0].startPointX += 10;
        movedLines[0].pt += Point2f(10, 0);
        movedLines[1].endPointX += 10;
        movedLines[1].startPointX += 10;
        movedLines[1].pt += Point2f(10, 0);

        Segment movedSegment(movedLines[0]);
        Segment temp(movedLines[1]);
        movedSegment.join(temp);

        SegmentMatch match;
        match.segment1 = segmentsVecAns2_[0];
        match.segment2 = movedSegment;
        match.computeOffsets();

        EXPECT_EQ(Point2f(10, 0), match.positionOffset);
    }

    TEST_F(SegmentMatchTest, computeOffsetsPositionValidity) {
        
        EXPECT_STREQ("NOT_IMPLEMENTED", "TODO");
    }

    TEST_F(SegmentMatchTest, computeOffsetsCombinedPerfect) {
        SegmentMatch match;
        match.segment1 = segmentsVecAutogen4_[0];
        match.segment2 = segmentsVecAutogen5_[0];

        match.computeOffsets();

        EXPECT_NEAR(M_PI_4, match.angleOffset, 1e-5);
        
        Point2f pt1 = match.segment1.data().front().pt;
        Point2f pt2 = match.segment2.data().front().pt;
        Point2f expectedOffset = pt2 - pt1;

        EXPECT_NEAR(expectedOffset.x, match.positionOffset.x, 1.0);
        EXPECT_NEAR(expectedOffset.y, match.positionOffset.y, 1.0);

    }

    TEST_F(SegmentMatchTest, computeOffsetsMirrored) {
        Segment seg1, seg2;
        seg1 = segmentsVecAutogen3_[0];
        seg2 = Segment(segmentsVecAutogen4_[0], 2, 1);

        SegmentMatch match;
        match.segment1 = seg1;
        match.segment2 = seg2;

        LmStatus status = match.computeOffsets();

        EXPECT_EQ(LM_STATUS_OK, status);
        EXPECT_NEAR(0, match.angleOffset, 1e-5);
        EXPECT_EQ(true, match.isFlipped);
    }

    void drawMatches(vector<SegmentMatch> matches, InputOutputArray img) {
        Segments matchedSegments;
        for (auto matchItr = matches.cbegin(); matchItr != matches.cend(); matchItr++) {
            const SegmentMatch& match = *matchItr;
            matchedSegments.addSegment(match.segment1);
        }
        matchedSegments.draw(img);
    }
    
    TEST_F(SegmentIntegrationTest, matchSectionToLongWall) {
        // Matches an L shaped section to a 4 line section forming a staircase
        Segments section, wall;
        LmStatus status;
        status = section.addLines(lines3_);
        EXPECT_EQ(LM_STATUS_OK, status);

        status = wall.addLines(lines4_);
        EXPECT_EQ(LM_STATUS_OK, status);

        vector<SegmentMatch> matches;
        status = wall.matchSegments(section, matches);
        EXPECT_EQ(LM_STATUS_OK, status);

        srand(time(NULL));

        Mat img;
        cvtColor(testImg4_, img, COLOR_GRAY2BGR);
        
        drawMatches(matches, img);

        bool imwriteSuccess = imwrite("debug/matchSectionToLongWall2.jpg", img);
        // Check matches
        ASSERT_EQ(3, matches.size());

        // match0 is bottom right, match1 is top left, match2 is flipped middle
        Segments answer1, answer2, answer3;
        answer1.addLines(KeyLines(lines4_.begin() + 0, lines4_.begin() + 2));
        answer2.addLines(KeyLines(lines4_.begin() + 1, lines4_.begin() + 3));
        answer3.addLines(KeyLines(lines4_.begin() + 2, lines4_.begin() + 4));
        
        EXPECT_SEGMENT_EQUAL(*answer1.data().front(), matches[1].segment1);
        EXPECT_SEGMENT_EQUAL(*answer2.data().front(), matches[2].segment1);
        EXPECT_SEGMENT_EQUAL(*answer3.data().front(), matches[0].segment1);

        EXPECT_NEAR(0, matches[0].angleOffset, 1e-4);
        EXPECT_NEAR(0, matches[1].angleOffset, 1e-4);
        EXPECT_NEAR(0, matches[2].angleOffset, 1e-4);

        EXPECT_EQ(false, matches[0].isFlipped);
        EXPECT_EQ(false, matches[1].isFlipped);
        EXPECT_EQ(true, matches[2].isFlipped);

        EXPECT_EQ(true, imwriteSuccess);
    }

    TEST_F(SegmentIntegrationTest, matchSectionToLongWallRot45) {
        // Matches an L shaped section to a 4 line section forming a staircase
        Segments section, wall;
        LmStatus status;
        status = section.addLines(lines3_);
        EXPECT_EQ(LM_STATUS_OK, status);

        status = wall.addLines(lines5_);
        EXPECT_EQ(LM_STATUS_OK, status);

        vector<SegmentMatch> matches;
        status = wall.matchSegments(section, matches);
        EXPECT_EQ(LM_STATUS_OK, status);

        srand(time(NULL));

        Mat img;
        cvtColor(testImg5_, img, COLOR_GRAY2BGR);
        
        drawMatches(matches, img);

        bool imwriteSuccess = imwrite("debug/matchSectionToLongWallRot45.jpg", img);
        // Check matches
        ASSERT_EQ(3, matches.size());

        Segments answer1, answer2, answer3;
        answer1.addLines(KeyLines(lines5_.begin() + 0, lines5_.begin() + 2));
        answer2.addLines(KeyLines(lines5_.begin() + 1, lines5_.begin() + 3));
        answer3.addLines(KeyLines(lines5_.begin() + 2, lines5_.begin() + 4));
        
        EXPECT_SEGMENT_EQUAL(*answer1.data().front(), matches[1].segment1);
        EXPECT_SEGMENT_EQUAL(*answer2.data().front(), matches[2].segment1);
        EXPECT_SEGMENT_EQUAL(*answer3.data().front(), matches[0].segment1);

        EXPECT_NEAR(-M_PI_4, matches[0].angleOffset, 1e-4);
        EXPECT_NEAR(-M_PI_4, matches[1].angleOffset, 1e-4);
        EXPECT_NEAR(-M_PI_4, matches[2].angleOffset, 1e-4);

        EXPECT_EQ(false, matches[0].isFlipped);
        EXPECT_EQ(false, matches[1].isFlipped);
        EXPECT_EQ(true, matches[2].isFlipped);

        EXPECT_EQ(true, imwriteSuccess);
    }
    
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}