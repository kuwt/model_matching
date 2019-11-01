#include <vector>
#include "rgbd.hpp"
#include "imageFileIO.h"
static float class_threshold = 0.30; // Cut-off probability

// camera parameters
/**************** ycb ***************/
//static std::vector<float> cam_intrinsics = { 1066.778, 312.986, 1067.487, 241.310 }; //YCB
//static float depth_scale = 1 / 10000.0f;
/**** ***********line mod *************/
//static std::vector<float> cam_intrinsics = { 572.4114, 325.2611, 573.57043, 242.04899 };
//static float depth_scale = 1 / 1000.0f;
/**** ***********packed *************/
static std::vector<float> cam_intrinsics = { 615.957763671875, 308.1098937988281, 615.9578247070312, 246.33352661132812 };
static float depth_scale = 1 / 8000.0f;
static int image_width = 640;
static int image_height = 480;


void convetXYZIsValidMap(std::string scene_path, std::string object_name)
{
	std::string rgb_location = scene_path + "/rgb.png";
	std::string depth_location = scene_path + "/depth.png";
	std::string class_probability_map_location = scene_path + "/probability_maps/" + object_name + ".png";

	/***********  load scene in sample form ********************/
	cv::Mat rgb_image;
	rgb_image = cv::imread(rgb_location, CV_LOAD_IMAGE_COLOR);

	cv::Mat depth_image;
	depth_image = cv::imread(depth_location, CV_16UC1);

	cv::Mat class_probability_map;
	class_probability_map = cv::imread(class_probability_map_location, CV_16UC1);

	cv::Mat XMap = cv::Mat(depth_image.size(),CV_32FC1);
	cv::Mat YMap = cv::Mat(depth_image.size(), CV_32FC1);
	cv::Mat ZMap = cv::Mat(depth_image.size(), CV_32FC1);
	cv::Mat isValidMap = cv::Mat::zeros(depth_image.size(), CV_8UC1);
	cv::Mat SegmentationMap = cv::Mat::zeros(depth_image.size(), CV_8UC1);
	/************** compute point cloud from depth image *******************/
	for (int i = 0; i < depth_image.rows; i++)
	{
		for (int j = 0; j < depth_image.cols; j++)
		{
			float depth = (float)depth_image.at<unsigned short>(i, j)*depth_scale;

			XMap.at<float>(i, j) = (float)((j - cam_intrinsics[1]) * depth / cam_intrinsics[0]) * 1000; // to mm
			YMap.at<float>(i, j) = (float)((i - cam_intrinsics[3]) * depth / cam_intrinsics[2]) * 1000;
			ZMap.at<float>(i, j) = depth * 1000;
		}
	}

	for (int i = 0; i < class_probability_map.rows; i++)
	{
		for (int j = 0; j < class_probability_map.cols; j++)
		{
			float class_probability = (float)class_probability_map.at<unsigned short>(i, j)*(1.0 / 10000);

			if (class_probability > class_threshold)
			{
				isValidMap.at<unsigned char>(i,j) = 255;
				SegmentationMap.at<unsigned char>(i, j) = 255;
			}
		}
	}

	imageFileIO::FILE_SaveImageTiffR(XMap, "./XMap.tif");
	imageFileIO::FILE_SaveImageTiffR(YMap, "./YMap.tif");
	imageFileIO::FILE_SaveImageTiffR(ZMap, "./ZMap.tif");
	cv::imwrite("./isValid.bmp", isValidMap);
	cv::imwrite("./segmentation.bmp", SegmentationMap);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (int i = 0; i < XMap.rows; i++)
	{
		for (int j = 0; j < XMap.cols; j++)
		{
			pcl::PointXYZRGB pt;
			pt.x = XMap.at<float>(i, j);
			pt.y = YMap.at<float>(i, j);
			pt.z = ZMap.at<float>(i, j);

			cv::Vec3b rgb_val = rgb_image.at<cv::Vec3b>(i, j);
			uint32_t rgb = ((uint32_t)rgb_val.val[2] << 16 | (uint32_t)rgb_val.val[1] << 8 | (uint32_t)rgb_val.val[0]);
			pt.rgb = *reinterpret_cast<float*>(&rgb);

			cloud->points.push_back(pt);
		}
	}
	pcl::io::savePLYFile("./pointcloud.ply", *cloud);

}