#include <gtest/gtest.h>

#include "location_matcher/line_detector.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {
    // Tolerance parameters
    const float LENGTH_TOLERANCE = 5;
    const float ANGLE_TOLERANCE = M_PI * 10/180;
    const float POSITION_TOLERANCE = 3;

    KeyLine getKeyLine(float startX, float startY, float endX, float endY) {
        KeyLine kl;

        kl.startPointX  = startX;
        kl.startPointY  = startY;
        kl.endPointX    = endX;
        kl.endPointY    = endY;
        Point2f klVec = kl.getEndPoint() - kl.getStartPoint();
        kl.angle        = atan2(klVec.y, klVec.x);
        kl.lineLength   = sqrt(klVec.dot(klVec));

        return kl;
    }

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

    // === LineDetector testing util functions ===
    bool KeyLineCompare(const KeyLine& line1, const KeyLine& line2) {
        return line1.lineLength < line2.lineLength;
    }

    void EXPECT_KEYLINE_EQUAL(const KeyLine& expected, const KeyLine& actual) {
        EXPECT_LE(LENGTH_TOLERANCE, abs(expected.lineLength - actual.lineLength));
        EXPECT_LE(ANGLE_TOLERANCE, abs(expected.angle - actual.angle));
        
        Point2f startDiff = expected.getStartPoint() - actual.getStartPoint();
        Point2f endDiff = expected.getEndPoint() - actual.getEndPoint();

        float startOffset = sqrt(startDiff.dot(startDiff));
        float endOffset = sqrt(endDiff.dot(endDiff));

        EXPECT_LE(POSITION_TOLERANCE, startOffset);
        EXPECT_LE(POSITION_TOLERANCE, endOffset);
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