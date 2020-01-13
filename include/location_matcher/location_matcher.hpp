#pragma once
#include "location_matcher/core.hpp"
#include "location_matcher/line_detector.hpp"
#include "location_matcher/line_filter.hpp"

namespace lm {

    struct Blueprint {
        std::string name;
        cv::Mat blueprintImg;
        cv::Point2f centroid;
        float scale;        // metres per pixel (unused ATM)
    };

    struct LocationMatch {
        std::string name;
        cv::Point2f position;   // pixel position of the blueprint's centroid in the search image
        double certainty;
    };

    class LocationMatcher {
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
        
        LmStatus addBlueprint(const Blueprint& blueprint);

        std::map<std::string, Blueprint> blueprints_;

        LineDetector lineDetector_;
        LineFilter lineFilter_;

        private:
        
    };
};