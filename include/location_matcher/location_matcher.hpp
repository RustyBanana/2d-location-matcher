#pragma once
#include "location_matcher/core.hpp"
#include "location_matcher/line_detector.hpp"
#include "location_matcher/line_filter.hpp"
#include "location_matcher/segment.hpp"
#include "location_matcher/utils.hpp"

namespace lm {

    struct Blueprint {
        std::string name;       // Identifier
        cv::Mat blueprintImg;   // Lines are extracted from this image to match against the map
        cv::Point2f centroid;   // Centroid of blueprintImg
        float scale;            // metres per pixel (unimplemented currently)
    };

    struct LocationMatch {
        std::string name;       // Identifier of blueprint it is matched against
        cv::Point2f position;   // pixel position of the blueprint's centroid in the search image
        double angle;           // Angle by which the blueprint is rotated to fit onto the found match
        double certainty;       // [0, 1] confidence in match (unimplemented currently)
    };

    /*  Typical usage for LocationMatcher
        // Create blueprints
        Blueprint bp1;
        bp1.name = "Long Section";
        bp1.blueprintImg = imread("path/to/image.jpg", IMREAD_GRAYSCALE);
        bp1.centroid = Point2f(50, 70);
        bp1.scale = 0.05;

        // Add blueprint to matcher
        LocationMatcher matcher;
        matcher.addBlueprint(bp1);

        // Create storage for matches and find matches in image
        vector<LocationMatch> matches;
        matcher.findMatch(map, matches);
    */

    class LocationMatcherTest;
    class LocationMatcher {
        friend class LocationMatcherTest;
        public:
        LocationMatcher();
        ~LocationMatcher();

        /*
            General Workflow:
            
            Initialization
                - A set of blueprints is passed to this function and stored by
                name. Each blueprint is an image of the a location to be
                searched for with an arbitary centroid provided. The centroid 
                is used as a reference point when a match is returned.

            Running:
                - An occupancy map is passed to this function in the form of an 
                image.
                - Lines are extracted from the image.
                - Each line is thickened and is used as a mask on the image to 
                remove any noise which does not lie on a line.
                - cv2 MatchShapes is used on the contours or the direct image 
                to find matches
                - The centroid of the matches are used to find the equivalent 
                location in the new image.

        */

        // Input: occupancy map as a 2D matrix with values closer to 0 as occupied, and 255 as free
        LmStatus findMatch(const cv::Mat& imageIn, std::vector<LocationMatch>& matchesOut);

        // Input: mask of 255 in areas which have been explored and hence cannot contain a line, and 0 in unexplored areas or obstructed areas
        LmStatus filterMatches(const cv::Mat& mask, std::vector<LocationMatch>& matchesInOut);
        
        // Add blueprint to blueprints_
        LmStatus addBlueprint(const Blueprint& blueprint);

        // Draw matches on to img for debugging purposes
        void drawMatch(cv::Mat img, const LocationMatch& match)  const;

        std::map<std::string, Blueprint> blueprints_;

        LineDetector lineDetector_;
        LineFilter lineFilter_;

        // Helper function for converting from SegmentMatch used by Segment to a LocationMatch
        LocationMatch segmentMatchToLocationMatch(const Blueprint& blueprint, const SegmentMatch& match) const;

        protected:
        
        
    };
};