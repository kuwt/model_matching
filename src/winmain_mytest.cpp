#include "rgbd.hpp"
#include <filesystem>

namespace fs = std::experimental::filesystem;
static std::string repo_path = "D:/ronaldwork/model_matching";

// rgbd parameters
static float voxel_size = 0.005; // In m
static float distance_threshold = 0.005; // for Congruent Set Matching and LCP computation
static int ppf_tr_discretization = 5; // In mm
static int ppf_rot_discretization = 5; // degrees
static float class_threshold = 0.30; // Cut-off probability

// camera parameters
static std::vector<float> cam_intrinsics = { 1066.778, 312.986, 1067.487, 241.310 }; //YCB
static float depth_scale = 1 / 10000.0f;

static int image_width = 640;
static int image_height = 480;

int testBruteforceReg(std::string scene_path, std::string object_name)
{
	std::string rgb_location = scene_path + "/rgb.png";
	std::string depth_location = scene_path + "/depth.png";
	std::string class_probability_map_location = scene_path + "/probability_maps/" + object_name + ".png";
	std::string model_map_path = repo_path + "/models/" + object_name + "/ppf_map";
	std::string model_location = repo_path + "/models/" + object_name + "/model_search.ply";
	std::string debug_path = scene_path + "/dbg";
	/***********  debug path ********************/
	fs::create_directories(debug_path);

	/***********  load PPF map ********************/
	PPFMapType model_map;
	rgbd::load_ppf_map(model_map_path, model_map);
	
	/********** load model ********************************/
	std::vector<Point3D> point3d_model;
	PCLPointCloud::Ptr cloud(new PCLPointCloud);
	pcl::io::loadPLYFile(model_location, *cloud);
	rgbd::load_ply_model(cloud, point3d_model, 1.0f);

	std::cout << "|M| = " << point3d_model.size() << ",  |map(M)| = " << model_map.size() << std::endl;

	/***********  load scene in sample form ********************/
	cv::Mat dummy_map = cv::Mat::zeros(image_height, image_width, CV_8UC1);
	std::vector<Point3D> point3d_scene;
	rgbd::load_rgbd_data_sampled(
		rgb_location,
		depth_location,
		class_probability_map_location,
		dummy_map,
		cam_intrinsics,
		depth_scale,
		voxel_size,
		class_threshold,
		point3d_scene);


	rgbd::save_as_ply(debug_path + "/sampled_scene.ply", point3d_scene, 1.0);
	std::cout << "|S| = " << point3d_scene.size() << std::endl;


	/***********  calculate ppf pairs********************/
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::pair<int,int>> goodPairs;
	for (int i = 0; i < point3d_scene.size(); i++)
	{
		for (int j = i + 1; j < point3d_scene.size(); j++)
		{
			float distThd = 0.050; // m
			Point3D p1 = point3d_scene[i];
			Point3D p2 = point3d_scene[j];

			float dist = std::sqrt(
				(p1.x() - p2.x()) * (p1.x() - p2.x())
				+ (p1.y() - p2.y()) * (p1.y() - p2.y())
				+ (p1.z() - p2.z()) * (p1.z() - p2.z()));

			if (dist < distThd)
			{
				//std::cout << "Trial " << i * point3d_scene.size() + j << " ,dist = " << dist << " < " << distThd << "\n";
				continue;
			}
			//std::cout << "Trial " << i * point3d_scene.size() + j << " ,dist = " << dist << " > " << distThd << "\n";
			goodPairs.push_back(std::make_pair(i, j));
		}
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << "calculate ppf pairs  " << goodPairs.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";
	/***********  shuffle ********************/
	int goodPairsMax = 2000;
	start = std::chrono::high_resolution_clock::now();
	std::random_shuffle(goodPairs.begin(), goodPairs.end());
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "random_shuffle " << goodPairs.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	/***********  calculate correspondences********************/
	start = std::chrono::high_resolution_clock::now();
	struct poseEst
	{
		int srcId[2];
		int tarId[2];
		int ppf[4];
		float transform[16];
		//Point3D srcCoord[2];
		//Point3D tarCoord[2];
	};

	std::vector<poseEst> poseEsts;
	for (int i = 0; i < goodPairs.size() && i < goodPairsMax; i++)
	{
		int firstIdx = goodPairs.at(i).first;
		int secondIdx = goodPairs.at(i).second;

		std::vector<int> ppf;
		rgbd::ppf_compute(
			point3d_scene[firstIdx],
			point3d_scene[secondIdx],
			ppf_tr_discretization,
			ppf_rot_discretization,
			ppf);

		auto map_it = model_map.find(ppf);
		if (map_it != model_map.end())
		{
			std::vector<std::pair<int, int> > v = map_it->second;
			for (int k = 0; k < v.size(); k++)
			{
				poseEst e;
				e.srcId[0] = firstIdx;
				e.srcId[1] = secondIdx;
				e.tarId[0] = v[k].first;
				e.tarId[1] = v[k].second;
				e.ppf[0] = ppf[0];
				e.ppf[1] = ppf[1];
				e.ppf[2] = ppf[2];
				e.ppf[3] = ppf[3];
				//e.srcCoord[0] = p1;
				//e.srcCoord[1] = p2;
				//e.tarCoord[0] = point3d_model[e.tarId[0]];
				//e.tarCoord[1] = point3d_model[e.tarId[1]];
				poseEsts.push_back(e);
			}
		}
	}

	finish = std::chrono::high_resolution_clock::now();
	std::cout << "calculate ppf pairs and correspondences " << poseEsts.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	// save readable format
	{
		std::string locationReadable;
		locationReadable = debug_path + "/ppfCorrespondences.txt";
		std::FILE* f = std::fopen(locationReadable.c_str(), "w");
		for (int i = 0; i < poseEsts.size(); ++i)
		{
			std::fprintf(f, "%d %d %d %d : <%d,%d> : <%d,%d> \n", 
				poseEsts.at(i).ppf[0],
				poseEsts.at(i).ppf[1],
				poseEsts.at(i).ppf[2],
				poseEsts.at(i).ppf[3],
				poseEsts.at(i).srcId[0],
				poseEsts.at(i).srcId[1],
				poseEsts.at(i).tarId[0],
				poseEsts.at(i).tarId[1]);
		}
		std::fclose(f);
	}


	/***********  calculate pose for each ppf correspondences ********************/







	/***********  verify pose  ********************/


	return 0;
}
