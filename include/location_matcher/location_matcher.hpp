#pragma once
#include "location_matcher/core.hpp"

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
        std::map<std::string, Blueprint> blueprints_;

        private:

    };
};