#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

#include "test/test_datasets.hpp"
#include "location_matcher/utils.hpp"
#include "location_matcher/location_matcher.hpp"

using namespace testing;
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {
    class LocationMatcherTest : public BaseTest {
        public:
        void SetUp() override {
            BaseTest::SetUp();

            bp4_.blueprintImg = testImg3_;
            bp4_.centroid = Point2f(48, 56);
            bp4_.name = "L section";
            bp4_.scale = 0.05;

            
        }

        LocationMatcher matcher;

        Blueprint bp4_;
    };

    void EXPECT_EQ_LOCATION_MATCH(const LocationMatch& ans, const LocationMatch& test) {
        EXPECT_EQ(ans.name, test.name);
        EXPECT_EQ(ans.angle, test.angle);
        EXPECT_EQ(ans.position.x, test.position.x);
        EXPECT_EQ(ans.position.y, test.position.y);
    }

    TEST_F(LocationMatcherTest, matchLToLongWall) {
        Blueprint bp1;
        bp1.blueprintImg = testImg3_;
        bp1.centroid = Point2f(48, 56);
        bp1.name = "L section";
        bp1.scale = 0.05;

        matcher.addBlueprint(bp1);

        vector<LocationMatch> matches;
        matcher.findMatch(testImg4_, matches);

        Mat img = testImg4_.clone();

        for (auto matchItr = matches.cbegin(); matchItr != matches.cend(); matchItr++) {
            matcher.drawMatch(img, *matchItr);
        }

        imwrite("debug/LocationMatcher_matchLToLongWall", img);
    }

    TEST_F(LocationMatcherTest, segmentMatchToLocationMatch) {
        Segments map, bp;
        EXPECT_EQ(LM_STATUS_OK, map.addLines(lines4_));
        EXPECT_EQ(LM_STATUS_OK, bp.addLines(lines3_));

        vector<SegmentMatch> matches;
        map.matchSegments(bp, matches);

        vector<LocationMatch> locationMatches;

        ASSERT_EQ(3, matches.size());

        for (int i = 0; i < 3; i++) {
            locationMatches.push_back(matcher.segmentMatchToLocationMatch(bp4_, matches[i]));
        }

        vector<LocationMatch> ansLocationMatches;
        LocationMatch lm1, lm2, lm3;
        lm1.name = bp4_.name;
        lm2.name = bp4_.name;
        lm3.name = bp4_.name;
        lm1.position = Point2f(48, 56);
        lm2.position = Point2f(48, 105);
        lm3.position = Point2f(116,105);
        lm1.angle = 0;
        lm2.angle = M_PI;
        lm3.angle = 0;

        ansLocationMatches.push_back(lm1);
        ansLocationMatches.push_back(lm2);
        ansLocationMatches.push_back(lm3);

        for (int i = 0; i < 3; i++) {
            EXPECT_EQ_LOCATION_MATCH(ansLocationMatches[i], locationMatches[i]);
        }
    }

    
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
