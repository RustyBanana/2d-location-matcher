#include <gtest/gtest.h>

#include "location_matcher/utils.hpp"
#include "location_matcher/line_detector.hpp"

#include "test/test_datasets.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {
    // Tolerance parameters

    class LineDetectorTest : public ::testing::Test{
        public:
        LineDetector *ld;

        Mat testImg1_;
        Mat testImg2_;

        // The actual lines in each image
        KeyLines lines1_;
        KeyLines lines2_;

        void SetUp() override {
            ld = new LineDetector();

            testImg1_ = imread("test/line-detector-test-1.jpg", IMREAD_GRAYSCALE);
            testImg2_ = imread("test/line-detector-test-2.jpg", IMREAD_GRAYSCALE);

            lines1_.push_back(getKeyLine(14, 33, 47, 33));
            lines1_.push_back(getKeyLine(72, 44, 72 ,77));

            lines2_.push_back(getKeyLine(21, 33, 57, 69));
            lines2_.push_back(getKeyLine(21, 69, 57, 69));
            lines2_.push_back(getKeyLine(82, 12, 82, 51));
        }
    };


    // === LINE DETECTOR TESTS ===
    TEST_F(LineDetectorTest, unconnectedLinesUnmasked) {

        KeyLines lines;
        ld->detect(testImg1_, lines);

        EXPECT_KEYLINES_EQUAL(lines1_, lines);
    }

    TEST_F(LineDetectorTest, unconnectedLinesMasked){
        Mat mask = Mat::zeros(testImg1_.size(), CV_8UC1);
        Mat roi = mask(Range::all(), cv::Range(0, 50));
        roi = 1;

        KeyLines lines;
        ld->detect(testImg1_, lines, mask);

        EXPECT_KEYLINES_EQUAL(KeyLines(lines1_.begin(), lines1_.begin() + 1), lines);
    }

    TEST_F(LineDetectorTest, connectedLinesUnmasked){
        KeyLines lines;
        ld->detect(testImg2_, lines);

        EXPECT_KEYLINES_EQUAL(lines1_, lines);      
    }

}   // namespace lm

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}