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

            bp3_.blueprintImg = testImg3_;
            bp3_.centroid = Point2f(56,48);
            bp3_.name = "L section";
            bp3_.scale = 0.05;

            
        }

        LocationMatcher matcher;

        Blueprint bp3_;
    };

    void EXPECT_EQ_LOCATION_MATCH(const LocationMatch& ans, const LocationMatch& test) {
        EXPECT_EQ(ans.name, test.name);
        EXPECT_NEAR(ans.angle, test.angle, M_PI*5/180);
        EXPECT_NEAR(ans.position.x, test.position.x, 3.0);
        EXPECT_NEAR(ans.position.y, test.position.y, 3.0
        );
    }

    TEST_F(LocationMatcherTest, matchLToLongWall) {

        matcher.addBlueprint(bp3_);

        vector<LocationMatch> matches;
        matcher.findMatch(testImg4_, matches);

        Mat img = testImg4_.clone();

        for (auto matchItr = matches.cbegin(); matchItr != matches.cend(); matchItr++) {
            matcher.drawMatch(img, *matchItr);
        }

        imwrite("debug/LocationMatcher_matchLToLongWall.jpg", img);
    }

    TEST_F(LocationMatcherTest, matchLToLongWallRot45) {

        matcher.addBlueprint(bp3_);

        vector<LocationMatch> matches;
        matcher.findMatch(testImg5_, matches);

        Mat img = testImg5_.clone();

        for (auto matchItr = matches.cbegin(); matchItr != matches.cend(); matchItr++) {
            matcher.drawMatch(img, *matchItr);
        }

        imwrite("debug/LocationMatcher_matchLToLongWallRot45.jpg", img);
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
            locationMatches.push_back(matcher.segmentMatchToLocationMatch(bp3_, matches[i]));
        }

        vector<LocationMatch> ansLocationMatches;
        LocationMatch lm1, lm2, lm3;
        lm1.name = bp3_.name;
        lm2.name = bp3_.name;
        lm3.name = bp3_.name;
        lm1.position = Point2f(56, 48);
        lm2.position = Point2f(56, 105);
        lm3.position = Point2f(116, 105);
        lm1.angle = 0;
        lm2.angle = M_PI;
        lm3.angle = 0;

        ansLocationMatches.push_back(lm3);
        ansLocationMatches.push_back(lm1);
        ansLocationMatches.push_back(lm2);

        for (int i = 0; i < 3; i++) {
            EXPECT_EQ_LOCATION_MATCH(ansLocationMatches[i], locationMatches[i]);
        }

        Mat img = testImg4_.clone();
        matcher.addBlueprint(bp3_);
        for (auto matchItr = locationMatches.cbegin(); matchItr != locationMatches.cend(); matchItr++) {
            matcher.drawMatch(img, *matchItr);
        }
        imwrite("debug/LocationMatcher_segmentMatchToLocationMatch.jpg", img);

    }

    
}

int main(int argc, char* argv[]) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
