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

    
    
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}