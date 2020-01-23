#include "location_matcher/location_matcher.hpp"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

namespace lm {

    LocationMatcher::LocationMatcher() {

    }

    LocationMatcher::~LocationMatcher() {

    }

    LmStatus LocationMatcher::findMatch(const cv::Mat& imageIn, std::vector<LocationMatch>& matchesOut) {
        // Extract the segments from the image
        KeyLines lines;
        lineDetector_.detect(imageIn, lines);
        Segments imageSegments;
        imageSegments.addLines(lines);

        // Compare each segment extracted from the blueprints to the segments from the image
        for (auto blueprintItr = blueprints_.cbegin(); blueprintItr != blueprints_.cend(); blueprintItr++) {
            Segments blueprintSegments;
            KeyLines blueprintLines;

            const Blueprint& blueprint = blueprintItr->second;

            lineDetector_.detect(blueprint.blueprintImg, blueprintLines);
            blueprintSegments.addLines(blueprintLines);

            // Extract matches between the two segments
            vector<SegmentMatch> matches;
            imageSegments.matchSegments(blueprintSegments, matches);

            for (auto matchItr = matches.cbegin(); matchItr != matches.cend(); matchItr++) {
                // Convert SegmentMatch format to LocationMatch format.
                LocationMatch locationMatch = segmentMatchToLocationMatch(blueprint, *matchItr);
                
                matchesOut.push_back(locationMatch);
            }
        }
        return LM_STATUS_OK;
    }

    LmStatus LocationMatcher::addBlueprint(const Blueprint& blueprint) {
        blueprints_[blueprint.name] = blueprint;
    }

    void LocationMatcher::drawMatch(cv::Mat img, const LocationMatch& match) const {
        const Blueprint& bp = blueprints_.at(match.name);

        // Draw the centroid in
        circle(img, match.position, 5, Scalar(50, 50, 50), 5);

        // Draw the segment in
        /*
        KeyLines lines;
        lineDetector_.detect(bp.blueprintImg, lines);
        Segments imageSegments;
        imageSegments.addLines(lines);

        imageSegments.draw(img);
        */
    }

    LocationMatch LocationMatcher::segmentMatchToLocationMatch(const Blueprint& blueprint, const SegmentMatch& match) const {
        // Convert SegmentMatch format to LocationMatch format.
        LocationMatch locationMatch;
        locationMatch.name = blueprint.name;
        locationMatch.certainty = match.confidence;
        locationMatch.angle = match.angleOffset;

        // Get position of centroid in image from blueprint match position and blueprint centroid
        KeyLine blueprintStartLine = match.segment2.data().front();
        KeyLine imageStartLine = match.segment1.data().front();
        Point2f lineToCentroid = blueprint.centroid - blueprintStartLine.pt;
        lineToCentroid = rotateVector(lineToCentroid, locationMatch.angle);
        Point2f centroid = imageStartLine.pt + lineToCentroid;
        locationMatch.position = centroid;

        return locationMatch;
    }
}
