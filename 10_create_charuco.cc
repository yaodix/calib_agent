#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include "aruco_samples_utility.hpp"


using namespace cv;

namespace {
const char* about = "Create a ChArUco board image";
//! [charuco_detect_board_keys]
const char* keys  =
        "{@outfile |res.png| Output image }"
        "{w        |  5    | Number of squares in X direction }"
        "{h        |  7    | Number of squares in Y direction }"
        "{sl       |  100  | Square side length (in millimeters) }"
        "{ml       |  60   | Marker side length (in millimeters) }"
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{cd       |       | Input file with custom dictionary }"
        "{m        |       | Margins size (in millimeters). Default is (squareLength-markerLength)/2 }"
        "{bb       | 1     | Number of bits in marker borders }"
        "{dpi      | 300   | Output image DPI for printable rasterization }"
        "{legacy   | true  | Use legacy ChArUco layout for compatibility with OpenCV samples/detection }"
        "{si       | false | show generated image }";
}
//! [charuco_detect_board_keys]

// usage: ./build/create_charucoboard -w 5 -h 4 -sl 150 -ml 112 -d 0 -m 20 -bb 1 -si true

namespace {

constexpr double kMillimetersPerInch = 25.4;
constexpr int kInfoBandHeight = 90;

int mmToPixels(float millimeters, int dpi) {
    return cvRound(static_cast<double>(millimeters) * dpi / kMillimetersPerInch);
}

const char* dictionaryName(int dictionaryId) {
    switch (dictionaryId) {
        case cv::aruco::DICT_4X4_50: return "DICT_4X4_50";
        case cv::aruco::DICT_4X4_100: return "DICT_4X4_100";
        case cv::aruco::DICT_4X4_250: return "DICT_4X4_250";
        case cv::aruco::DICT_4X4_1000: return "DICT_4X4_1000";
        case cv::aruco::DICT_5X5_50: return "DICT_5X5_50";
        case cv::aruco::DICT_5X5_100: return "DICT_5X5_100";
        case cv::aruco::DICT_5X5_250: return "DICT_5X5_250";
        case cv::aruco::DICT_5X5_1000: return "DICT_5X5_1000";
        case cv::aruco::DICT_6X6_50: return "DICT_6X6_50";
        case cv::aruco::DICT_6X6_100: return "DICT_6X6_100";
        case cv::aruco::DICT_6X6_250: return "DICT_6X6_250";
        case cv::aruco::DICT_6X6_1000: return "DICT_6X6_1000";
        case cv::aruco::DICT_7X7_50: return "DICT_7X7_50";
        case cv::aruco::DICT_7X7_100: return "DICT_7X7_100";
        case cv::aruco::DICT_7X7_250: return "DICT_7X7_250";
        case cv::aruco::DICT_7X7_1000: return "DICT_7X7_1000";
        case cv::aruco::DICT_ARUCO_ORIGINAL: return "DICT_ARUCO_ORIGINAL";
        default: return "custom";
    }
}

cv::Mat appendInfoBand(const cv::Mat& boardImage, const std::string& summary) {
    cv::Mat canvas(boardImage.rows + kInfoBandHeight, boardImage.cols, CV_8UC1, cv::Scalar(255));
    boardImage.copyTo(canvas.rowRange(0, boardImage.rows));
    cv::putText(canvas, summary, cv::Point(24, boardImage.rows + 52), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(0), 1, cv::LINE_AA);
    return canvas;
}

std::vector<std::string> normalizeArgs(int argc, char* argv[]) {
    std::vector<std::string> normalized;
    normalized.reserve(argc);
    for (int index = 0; index < argc; ++index) {
        std::string current = argv[index];
        if (index > 0 && !current.empty() && current[0] == '-' && current.find('=') == std::string::npos &&
            index + 1 < argc) {
            std::string next = argv[index + 1];
            if (!next.empty() && next[0] != '-') {
                normalized.push_back(current + "=" + next);
                ++index;
                continue;
            }
        }
        normalized.push_back(current);
    }
    return normalized;
}

} // namespace

int main(int argc, char *argv[]) {
    std::vector<std::string> normalizedArgs = normalizeArgs(argc, argv);
    std::vector<const char*> parserArgv;
    parserArgv.reserve(normalizedArgs.size());
    for (const std::string& arg : normalizedArgs) {
        parserArgv.push_back(arg.c_str());
    }

    CommandLineParser parser(static_cast<int>(parserArgv.size()), parserArgv.data(), keys);
    parser.about(about);
    if (argc == 1) {
        parser.printMessage();
    }

    int squaresX = parser.get<int>("w");
    int squaresY = parser.get<int>("h");
    float squareLengthMm = parser.get<float>("sl");
    float markerLengthMm = parser.get<float>("ml");
    float marginsMm = (squareLengthMm - markerLengthMm) * 0.5f;
    if(parser.has("m")) {
        marginsMm = parser.get<float>("m");
    }

    int borderBits = parser.get<int>("bb");
    int dpi = parser.get<int>("dpi");
    bool useLegacyPattern = parser.get<bool>("legacy");
    bool showImage = parser.get<bool>("si");

    std::string pathOutImg = parser.get<std::string>(0);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    if (squaresX < 2 || squaresY < 2) {
        std::cerr << "Board dimensions must be at least 2x2 squares." << std::endl;
        return 1;
    }
    if (squareLengthMm <= 0.0f || markerLengthMm <= 0.0f || markerLengthMm >= squareLengthMm) {
        std::cerr << "Require 0 < markerLength < squareLength." << std::endl;
        return 1;
    }
    if (marginsMm < 0.0f) {
        std::cerr << "Margins must be non-negative." << std::endl;
        return 1;
    }
    if (borderBits < 1) {
        std::cerr << "Border bits must be at least 1." << std::endl;
        return 1;
    }
    if (dpi <= 0) {
        std::cerr << "DPI must be positive." << std::endl;
        return 1;
    }

    //! [create_charucoBoard]
    aruco::Dictionary dictionary = readDictionatyFromCommandLine(parser);
    cv::aruco::CharucoBoard board(Size(squaresX, squaresY), squareLengthMm, markerLengthMm, dictionary);
    board.setLegacyPattern(useLegacyPattern);
    //! [create_charucoBoard]

    // show created board
    //! [generate_charucoBoard]
    Mat boardImage;
    Size imageSize;
    int squareLengthPx = mmToPixels(squareLengthMm, dpi);
    int marginsPx = mmToPixels(marginsMm, dpi);
    imageSize.width = squaresX * squareLengthPx + 2 * marginsPx;
    imageSize.height = squaresY * squareLengthPx + 2 * marginsPx;
    board.generateImage(imageSize, boardImage, marginsPx, borderBits);
    //! [generate_charucoBoard]

    int dictionaryId = parser.has("d") ? parser.get<int>("d") : cv::aruco::DICT_4X4_50;
    std::ostringstream summary;
    summary << squaresX << "x" << squaresY
            << " | Square: " << squareLengthMm << " mm"
            << " | Marker: " << markerLengthMm << " mm"
            << " | Margin: " << marginsMm << " mm"
            << " | DPI: " << dpi
            << " | Dictionary: ArUco " << dictionaryName(dictionaryId)
            << " | Legacy: " << (useLegacyPattern ? "on" : "off");
    boardImage = appendInfoBand(boardImage, summary.str());

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    if (pathOutImg != "")
        imwrite(pathOutImg, boardImage);
    return 0;
}