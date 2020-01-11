#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "location_matcher/utils.hpp"
#include "location_matcher/line_filter.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {

    class LineFilterTest : public ::testing::Test{
        public:
        LineFilter *lf;

        Mat testImg1_;
        Mat testImg2_;

        // The actual lines in each image
        KeyLines lines1_;
        KeyLines lines2_;

        void SetUp() override {
            lf = new LineFilter(30, 40);

            testImg1_ = imread("test/line-filter-test-1.jpg", IMREAD_GRAYSCALE);
            testImg2_ = imread("test/line-filter-test-2.jpg", IMREAD_GRAYSCALE);

            lines1_.push_back(getKeyLine(14, 33, 47, 33));
            lines1_.push_back(getKeyLine(72, 44, 72 ,77));

            lines2_.push_back(getKeyLine(21, 33, 57, 69));
            lines2_.push_back(getKeyLine(21, 69, 57, 69));
            lines2_.push_back(getKeyLine(82, 12, 82, 51));
        }
    };

    // No lines should be filtered because they should all be within the filter's range
    TEST_F(LineFilterTest, allLinesUnfiltered) {
        Mat filteredImg;
        LmStatus status = lf->filterByLine(testImg1_, filteredImg, lines1_, 5);

        Mat diff = testImg1_ != filteredImg;
        bool isEqual = countNonZero(diff) == 0;

        EXPECT_EQ(true, isEqual);
        EXPECT_EQ(LM_STATUS_OK, status);
    }

    TEST_F(LineFilterTest, allLinesFiltered) {
        Mat filteredImg;

        // When the range is smaller than the lines
        lf->setMinLineLength(10);
        lf->setMaxLineLength(20);

        LmStatus status = lf->filterByLine(testImg1_, filteredImg, lines1_, 5);

        Mat empty = Mat::ones(testImg1_.size(), CV_8UC1);
        Mat diff = empty != filteredImg;
        bool isEqual = countNonZero(diff) == 0;

        EXPECT_EQ(true, isEqual);
        EXPECT_EQ(LM_STATUS_OK, status);

        // When the range is larger than the lines
        lf->setMinLineLength(40);
        lf->setMaxLineLength(80);

        LmStatus status2 = lf->filterByLine(testImg1_, filteredImg, lines1_, 5);

        Mat empty2 = Mat::ones(testImg1_.size(), CV_8UC1);
        Mat diff2 = empty != filteredImg;
        bool isEqual2 = countNonZero(diff) == 0;

        EXPECT_EQ(true, isEqual2);
        EXPECT_EQ(LM_STATUS_OK, status2);
    }

    TEST_F(LineFilterTest, someLinesFiltered) {
        Mat filteredImg;
        Mat empty = Mat::ones(testImg1_.size(), CV_8UC1);

        LmStatus status = lf->filterByLine(testImg1_, filteredImg, lines1_, 5);

        Mat diffWithEmpty = empty != filteredImg;
        bool isEqualToEmpty = countNonZero(diffWithEmpty) == 0;

        Mat diffWithInput = testImg1_ != filteredImg;
        bool isEqualToInput = countNonZero(diffWithInput) == 0;

        EXPECT_EQ(false, isEqualToEmpty);
        EXPECT_EQ(false, isEqualToInput);
        EXPECT_EQ(LM_STATUS_OK, status);
    }
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}