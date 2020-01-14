#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "location_matcher/utils.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {
    const float LENGTH_TOLERANCE = 5;
    const float ANGLE_TOLERANCE = M_PI * 10/180;
    const float POSITION_TOLERANCE = 3;
    // === LineDetector testing util functions ===
    bool KeyLineCompare(const KeyLine& line1, const KeyLine& line2);

    void EXPECT_KEYLINE_EQUAL(const KeyLine& expected, const KeyLine& actual);

    void EXPECT_KEYLINES_EQUAL(const KeyLines& expected_, const KeyLines& actual_);

    class BaseTest : public ::testing::Test{
        public:

        Mat testImg1_;
        Mat testImg2_;
        Mat testImg3_;
        Mat testImg4_;
        Mat testImg5_;

        // The actual lines in each image
        KeyLines lines1_;
        KeyLines lines2_;
        KeyLines lines3_;
        KeyLines lines4_;
        KeyLines lines5_;
        

        void SetUp() override;
    };
}