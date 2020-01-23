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

    class LineDetectorTest : public BaseTest{
        public:
        LineDetector *ld;

        void SetUp() override {
            ld = new LineDetector();

            BaseTest::SetUp();
        }
    };


    // === LINE DETECTOR TESTS ===
    TEST_F(LineDetectorTest, unconnectedLinesUnmasked) {

        KeyLines lines;
        ld->detect(testImg1_, lines);

        Mat img = testImg1_.clone();
        drawLines(img, lines);
        bool imwriteSucess = imwrite("debug/LineDetectorTest_unconnectedLinesUnmasked.jpg", img);
        EXPECT_EQ(true, imwriteSucess);

        EXPECT_KEYLINES_EQUAL(lines1_, lines);
    }

    TEST_F(LineDetectorTest, unconnectedLinesMasked){
        Mat mask = Mat::zeros(testImg1_.size(), CV_8UC1);
        Mat roi = mask(Range::all(), cv::Range(0, 50));
        roi = 1;

        KeyLines lines;
        ld->detect(testImg1_, lines, mask);

        EXPECT_KEYLINES_EQUAL(KeyLines(lines1_.begin(), lines1_.begin() + 1), lines);

        Mat img = testImg1_.clone();
        drawLines(img, lines);
        bool imwriteSucess = imwrite("debug/LineDetectorTest_unconnectedLinesMasked.jpg", img);
        EXPECT_EQ(true, imwriteSucess);
    }

    TEST_F(LineDetectorTest, connectedLinesUnmasked){
        KeyLines lines;
        ld->detect(testImg2_, lines);

        EXPECT_KEYLINES_EQUAL(lines2_, lines);     

        Mat img = testImg2_.clone();
        drawLines(img, lines);
        bool imwriteSucess = imwrite("debug/LineDetectorTest_connectedLinesUnmasked.jpg", img);
        EXPECT_EQ(true, imwriteSucess); 
    }

    TEST_F(LineDetectorTest, longWall){
        vector<KeyLine> lines;
        ld->detect(testImg4_, lines);

        EXPECT_KEYLINES_EQUAL(lines4_, lines);     

        Mat img = testImg4_.clone();
        drawLines(img, lines);
        bool imwriteSucess = imwrite("debug/LineDetectorTest_longWall.jpg", img);
        EXPECT_EQ(true, imwriteSucess); 
    }

    TEST_F(LineDetectorTest, longWallRot45){
        KeyLines lines;
        ld->detect(testImg5_, lines);

        EXPECT_KEYLINES_EQUAL(lines5_, lines);     

        Mat img = testImg5_.clone();
        drawLines(img, lines);
        bool imwriteSucess = imwrite("debug/LineDetectorTest_longWallRot45.jpg", img);
        EXPECT_EQ(true, imwriteSucess); 
    }

}   // namespace lm

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}