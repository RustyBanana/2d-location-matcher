#pragma once

#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>

namespace lm {
    enum LmStatus {
        LM_STATUS_OK = 0,
        LM_STATUS_ERROR_GENERIC,
        LM_STATUS_ERROR_LINES_UNCONNECTED,
        LM_STATUS_ERROR_MATCH_FAILED,
        LM_STATUS_SIZE_MISMATCH
    };

    enum LineJoint {
        LINE_JOINT_NONE     = 0b000,
        LINE_JOINT_SS       = 0b001,
        LINE_JOINT_SE       = 0b011, 
        LINE_JOINT_ES       = 0b101,
        LINE_JOINT_EE       = 0b111
    };

    inline float dist(cv::Point2f pt1, cv::Point2f pt2) {
        cv::Point2f diff = pt2 - pt1;
        return sqrt(diff.dot(diff));
    }

    // Returns angle in the range [0, 2*pi] + offset
    inline float wrap2pi(float angle, float offset=0) {
        angle = fmod(angle + M_PI, 2*M_PI);
        if (angle < 0) {
            angle += 2 * M_PI;
        }
        return angle + offset;
    }

    // Returns angle in the range [0, pi] + offset
    inline float wrappi(float angle, float offset=0) {
        angle = wrap2pi(angle);
        return angle > M_PI ? angle - M_PI + offset: angle + offset;
    }

    inline float angleDiff(float angle1, float angle2, float wrapAngle = 2*M_PI) {
        float diff = angle1 - angle2;
        const float tolerance = 1e-5;
        if (abs(abs(diff) - wrapAngle/2) <= tolerance) {
            return wrapAngle/2; // Rounding off both -wrapAngle/2 and wrapAngle/2 to the same value.
        } else if (abs(diff) <= wrapAngle/2) {
            return diff;
        } else {
            return diff < 0 ? diff + wrapAngle : diff - wrapAngle;
        }
    }

    typedef std::vector<cv::line_descriptor::KeyLine> KeyLines;

    typedef const KeyLines& KeyLinesIn;
    typedef KeyLines& KeyLinesOut;

    // Each Keyline in the list is placed adjacent to the KeyLines they are connected to in the image.
    typedef std::list<std::shared_ptr<cv::line_descriptor::KeyLine>> Segment11;


};