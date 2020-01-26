#pragma once

#include <numeric> // accumulate
#include <time.h>   // For initializing random

#include "opencv2/core.hpp"
#include "opencv2/line_descriptor.hpp"

#include "location_matcher/core.hpp"
#include "location_matcher/utils.hpp"

namespace lm{
    // F = Front, B = Back
    enum SegmentJoint {
        SEGMENT_JOINT_NONE  = 0b000,
        SEGMENT_JOINT_FF    = 0b001,
        SEGMENT_JOINT_FB    = 0b011,
        SEGMENT_JOINT_BF    = 0b101,
        SEGMENT_JOINT_BB    = 0b111
    };
    const int SEGMENT_JOINT_1 = 2;
    const int SEGMENT_JOINT_2 = 1;
    const int SEGMENT_JOINT_JOINED = 0;

    class Segment;
    class Segments;
    struct SegmentMatch;
                        
    // Compares the likeness of two lines, with 1.0 being the same, 0 being completely different.
    float compareLines(const cv::line_descriptor::KeyLine& line1, const cv::line_descriptor::KeyLine& line2, bool angleInvariont=false);

    /*
        Segment contains a single sequence of connected lines. Branch is not allowed in Segment. I.e. there is no implementation for a single corner with 3 lines connected to it.
    */
    class Segment {
        friend class Segments;
        friend class SegmentMatch;
        friend class SegmentTest;
        
        // Data is stored as a list for quick reversing and accessing from the front and back
        typedef std::list<cv::line_descriptor::KeyLine> segment_t;

        public:
        Segment();

        // Create a segment starting from a single line
        Segment(const cv::line_descriptor::KeyLine& line);

        // Create a segment from slicing an existing segment.
        Segment(const Segment& segment, int beginIndex, int endIndex);

        const segment_t& data() const;

        // The other.data_ is appended to this->data_ and other.data_ is omptied
        LmStatus join(Segment& other);

        // Check if the line is joined from the Front(F) or Back(B) of segment1 to the F or B of segment2.
        SegmentJoint isJoinedTo(const Segment& other) const;

        // Returns the k nearest best matches
        LmStatus compareWith(const Segment& other, std::vector<SegmentMatch>& matches) const;

        // Draw the segment on to ImgIn for debug purposes. imgIn should be converted to BGR format.
        void draw(cv::InputOutputArray imgIn, cv::Scalar color, std::string label) const;


        protected:
        // Helper function for compareWith()
        // It traverses a diagonal from startIndex in the direction of incrementIndex and if it finds 2 or more adjacent values in likenessMatrix >= likenessRatio, it will add the SegmentMatch to matches
        void findMatchesInDiag(
            cv::Mat likenessMatrix, 
            cv::Point2i startIndex, 
            cv::Point2i incrementIndex, 
            float likenessThreshold, 
            const Segment& other, 
            std::vector<SegmentMatch>& matches) const;

        segment_t data_;
    };

    /*
        Contains a group of Segment. Used for cases such as storing all the Segment extracted from an image.
    */
    class Segments {
        friend class SegmentsTest;

        typedef std::vector<std::shared_ptr<Segment>> data_t;

        public:
        LmStatus addSegment(const Segment& segment);
        LmStatus addLines(const KeyLines& lines);
        LmStatus clear();

        // Return k nearest neighbour matches sorted by match strength.
        // Returns SegmentMatch with segment1 from this Segments, and segment2 from the other Segments
        LmStatus matchSegments(const Segment& segment, std::vector<SegmentMatch>& matches);
        
        LmStatus matchSegments(const Segments& segments, std::vector<SegmentMatch>& matches);

        // imgIn is expected to be a BGR image
        // Draws every Segment as a different colour, with each line in the segment labelled with the index number. This function is used for debugging pursposes.
        void draw(cv::InputOutputArray imgIn) const;

        const data_t& data() const;

        private:
        // Stores each segment as a shared_ptr because when joining segments some will be lost
        data_t data_;

        // Helper function to remove duplicates in data_
        void pruneSegments(std::vector<bool> keepList);
    };

    struct SegmentMatch {
        // StdDeviation thresholds for whether a line is considered to be a valid match. 
        static float angleThreshold;
        static float positionThreshold;
        
        SegmentMatch();
        SegmentMatch(Segment seg1, int startIndex1, int endIndex1, Segment seg2, int startIndex2, int endIndex2);

        Segment segment1;
        Segment segment2;
        int segment1Index[2];   // segment1 is constructed from copying the original segment from segment1Index[0] to segment1Index[1]
        int segment2Index[2];
        cv::Point2f positionOffset; // translation by which segment1 needs to be moved to match segment2
        float angleOffset;  // angle by which segment1 needs to be rotated to match segment2
        bool isFlipped;     // Unused and can be removed; didn't have time to remove.
        float confidence;   // [0,1] confidence in the match

        // Returns LM_STATUS_OK if the angleOffset and the positionOffset are within their thresholds. Returns LM_STATUS_ERROR_MATCH_FAILED if they are outside their thresholds.
        // Sets positionOffset and angleOffset
        LmStatus computeOffsets();
        LmStatus computeConfidence();   // Unimplemented
    };
};