#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

int main(int argc, char** argv) {
	const std::string imagePath = argc > 1 ? argv[1] : "../dict_44.png";
	cv::Mat image = cv::imread(imagePath);
	if (image.empty()) {
		std::cerr << "failed to load image: " << imagePath << std::endl;
		return 1;
	}

	// auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
	// cv::aruco::CharucoBoard board(cv::Size(5, 4), 0.04f, 0.032f, dictionary);
	// board.setLegacyPattern(true);
	// cv::aruco::CharucoDetector charucoDetector(board);
    int squaresX = 5;
    int squaresY = 4;
    float squareLength = 0.04f;
    float markerLength = 0.032f;
    bool refine = false;
    int camId = 0;
	 // create charuco board object
    aruco::CharucoBoard charucoBoard(Size(squaresX, squaresY), squareLength, markerLength, dictionary);
 		detectorParams = aruco::DetectorParameters::create(); // create default parameters for marker detection
		detectorP
    // create charuco detector
    aruco::CharucoParameters charucoParams;
    charucoParams.tryRefineMarkers = refine; // if tryRefineMarkers, refineDetectedMarkers() will be used in detectBoard()
    charucoParams.cameraMatrix = cv::Mat(); // cameraMatrix can be used in detectBoard()
    charucoParams.distCoeffs = cv::Mat(); // distCoeffs can be used in detectBoard()
    aruco::CharucoDetector charucoDetector(charucoBoard, charucoParams, detectorParams);

	std::vector<cv::Point2f> charucoCorners;
	std::vector<int> charucoIds;
	std::vector<std::vector<cv::Point2f>> markerCorners;
	std::vector<int> markerIds;

	charucoDetector.detectBoard(image, charucoCorners, charucoIds, markerCorners, markerIds);

	std::cout << "markerIds=" << markerIds.size()
			  << ", charucoIds=" << charucoIds.size() << std::endl;

	cv::Mat vis = image.clone();
	if (!markerIds.empty()) {
		cv::aruco::drawDetectedMarkers(vis, markerCorners, markerIds);
	}
	if (!charucoIds.empty()) {
		cv::aruco::drawDetectedCornersCharuco(vis, charucoCorners, charucoIds);
	}

	cv::imwrite("dict_55_detected.png", vis);
	return 0;
}