#include "test/test_datasets.hpp"

namespace lm{
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