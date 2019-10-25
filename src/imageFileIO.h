/* Copyright (C) ASTRI - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by kuwingto <ronaldku@astri.org>, Jan 2019
*/


#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class imageFileIO
{
public:
	static const int OK = 0;
	static const int ERR = -1;
	/*****************************
	* Load Image
	****************************/
	static cv::Mat loadImage(const std::string& name);

	static cv::Mat loadImage(const std::string& name, int mode);

	static int loadImage(const std::string& name, int mode, cv::Mat &image);

	static bool IsGrayScaleImage(const cv::Mat &image);

	/*****************************
	*	Load Tiff
	****************************/
	static int FILE_LoadImageTiffR(cv::Mat &matImage, std::string strFilename);
	/*****************************
	*	Save Tiff
	****************************/
	static int FILE_SaveImageTiffR(cv::Mat matImage, std::string strFilename);

	static int FILE_SaveImageTiffint(cv::Mat matImage, std::string strFilename);

	static int FileSaveImage_pverealto8bit(cv::Mat matImage, std::string strFilename);

	static int FileSaveImage_realto8bit(cv::Mat matImage, std::string strFilename);
};


