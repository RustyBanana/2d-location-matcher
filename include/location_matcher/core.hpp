#include <opencv2/core.hpp>
#include <opencv2/line_descriptor.hpp>

namespace lm {
    enum LmStatus {
        LM_STATUS_OK,
        LM_STATUS_ERROR_GENERIC
    };

    typedef std::vector<cv::line_descriptor::KeyLine> KeyLines;

    typedef const KeyLines& KeyLinesIn;
    typedef KeyLines& KeyLinesOut;

};