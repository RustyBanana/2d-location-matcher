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
                segments1_.push_back(Segment(lines1_[i]));
            }
            for (int i = 0; i < lines2_.size(); i++) {
                segments1_.push_back(Segment(lines2_[i]));
            }

        }

        vector<Segment> segments1_;
        vector<Segment> segments2_;
    };

    TEST_F(SegmentTest, isJoinedToUnconnected) {
        EXPECT_EQ(SEGMENT_JOINT_NONE, segments1_[0].isJoinedTo(segments1_[1]));
    }

    TEST_F(SegmentTest, isJoinedToConnected) {
        EXPECT_EQ(SEGMENT_JOINT_BB, segments2_[0].isJoinedTo(segments2_[1]));
    }
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}