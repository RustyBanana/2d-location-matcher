#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "location_matcher/utils.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {
    class BaseTest : public ::testing::Test{
        public:

        Mat testImg1_;
        Mat testImg2_;

        // The actual lines in each image
        KeyLines lines1_;
        KeyLines lines2_;

        void SetUp() override;
    };
}