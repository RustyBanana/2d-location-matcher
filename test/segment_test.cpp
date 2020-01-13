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
    class SegmentTest : public BaseTest {
        public:
        void SetUp() override {
            BaseTest::SetUp();
            
            for (int i = 0; i < lines1_.size(); i++) {
                segmentsVec1_.push_back(Segment(lines1_[i]));
            }
            for (int i = 0; i < lines2_.size(); i++) {
                segmentsVec2_.push_back(Segment(lines2_[i]));
            }

        }

        vector<Segment> segmentsVec1_;
        vector<Segment> segmentsVec2_;
    };

    class SegmentsTest : public SegmentTest {
        public:
        void SetUp() override {
            SegmentTest::SetUp();
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
        EXPECT_KEYLINE_EQUAL(pSegment->data().front(), segmentsVec1_[0].data().front());
        pSegment = segments.data()[0];
        EXPECT_KEYLINE_EQUAL(pSegment->data().back(), segmentsVec1_[1].data().front());
        pSegment = segments.data()[1];
        EXPECT_KEYLINE_EQUAL(pSegment->data().front(), segmentsVec1_[2].data().front());        
    }

    
    
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}