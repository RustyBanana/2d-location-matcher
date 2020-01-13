#include "test/test_datasets.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm{

    // === LineDetector testing util functions ===
    bool KeyLineCompare(const KeyLine& line1, const KeyLine& line2) {
        return line1.lineLength < line2.lineLength;
    }

    void EXPECT_KEYLINE_EQUAL(const KeyLine& expected, const KeyLine& actual) {
        EXPECT_GE(LENGTH_TOLERANCE, abs(expected.lineLength - actual.lineLength));
        EXPECT_GE(ANGLE_TOLERANCE, abs(expected.angle - actual.angle));
        
        Point2f startDiff = expected.getStartPoint() - actual.getStartPoint();
        Point2f endDiff = expected.getEndPoint() - actual.getEndPoint();

        float startOffset = sqrt(startDiff.dot(startDiff));
        float endOffset = sqrt(endDiff.dot(endDiff));

        EXPECT_GE(POSITION_TOLERANCE, startOffset);
        EXPECT_GE(POSITION_TOLERANCE, endOffset);
    }

    void EXPECT_KEYLINES_EQUAL(const KeyLines& expected_, const KeyLines& actual_) {
        KeyLines expected = expected_;
        KeyLines actual = actual_;
        sort(expected.begin(), expected.end(), KeyLineCompare);
        sort(actual.begin(), actual.end(), KeyLineCompare);

        ASSERT_EQ(expected.size(), actual.size());

        for (int i = 0; i < expected.size(); i++) {
            EXPECT_KEYLINE_EQUAL(expected[i], actual[i]);
        }
    }

    void BaseTest::SetUp() {
        testImg1_ = imread("test/line-detector-test-1.jpg", IMREAD_GRAYSCALE);
        testImg2_ = imread("test/line-detector-test-2.jpg", IMREAD_GRAYSCALE);

        lines1_.push_back(getKeyLine(14, 33, 47, 33));
        lines1_.push_back(getKeyLine(72, 44, 72 ,77));

        lines2_.push_back(getKeyLine(21, 33, 57, 69));
        lines2_.push_back(getKeyLine(21, 69, 57, 69));
        lines2_.push_back(getKeyLine(82, 12, 82, 51));
    }
}