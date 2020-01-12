#pragma once

#include "opencv2/core.hpp"
#include "opencv2/line_descriptor.hpp"

#include "location_matcher/core.hpp"

namespace lm{
    // F = Front, B = Back
    enum SegmentJoint {
        SEGMENT_JOINT_NONE  = 0b000,
        SEGMENT_JOINT_FF    = 0b001,
        SEGMENT_JOINT_FB    = 0b011,
        SEGMENT_JOINT_BF    = 0b101,
        SEGMENT_JOINT_BB    = 0b111
    };
    const int SEGMENT_JOINT_1 = 0b100;
    const int SEGMENT_JOINT_2 = 0b010;

    enum LineJoint {
        LINE_JOINT_NONE     = 0b000,
        LINE_JOINT_SS       = 0b001,
        LINE_JOINT_SE       = 0b011, 
        LINE_JOINT_ES       = 0b101,
        LINE_JOINT_EE       = 0b111
    };

    LineJoint isJoinedTo(const cv::line_descriptor::KeyLine& line1,
                         const cv::line_descriptor::KeyLine& line2,
                         float distThreshold);

    class Segments {
        public:
        LmStatus addLines(const KeyLines& lines);
        LmStatus clear();

        // Return k nearest neighbour matches sorted by match strength
        LmStatus matchSegment(const Segment& segment, typeTBD& matches);

        private:
        // Stores each segment as a shared_ptr because when joining segments some will be lost
        std::vector<std::shared_ptr<Segment>> data_;

        // Helper function to remove duplicates in data_
        void pruneSegments(std::vector<bool> keepList);
    };

    class Segment {
        public:
        Segment(const cv::line_descriptor::KeyLine& line);

        // The other.data_ is appended to this->data_ and other.data_ is omptied
        LmStatus join(Segment& other);
        SegmentJoint isJoinedTo(const Segment& other) const;

        private:
        std::list<cv::line_descriptor::KeyLine> data_;
    };
};