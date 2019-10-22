#include <filesystem>
#include "rgbd.hpp"
#include "accelerators/kdtree.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define FLANN_USE_CUDA
#include <flann/flann.hpp>

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


int gpulesscs(std::string scene_path, std::string object_name)
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
	int goodPairsMax = 200;
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
		Eigen::Matrix<float, 4, 4> trans44;
		float lcp;
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
			std::vector<std::pair<int, int> > &v = map_it->second;
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

	/***********  calculate transform for each ppf correspondences ********************/
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < poseEsts.size() ; ++i)
	{
		int src_id0 = poseEsts.at(i).srcId[0];
		int src_id1 = poseEsts.at(i).srcId[1];
		int tar_id0 = poseEsts.at(i).tarId[0];
		int tar_id1 = poseEsts.at(i).tarId[1];

		Point3D::VectorType src_p1 = point3d_model[src_id0].pos();
		Point3D::VectorType src_p2 = point3d_model[src_id1].pos();
		Point3D::VectorType src_n1 = point3d_model[src_id0].normal();
		Point3D::VectorType src_n2 = point3d_model[src_id1].normal();

		Point3D::VectorType tar_p1 = point3d_scene[tar_id0].pos();
		Point3D::VectorType tar_p2 = point3d_scene[tar_id1].pos();
		Point3D::VectorType tar_n1 = point3d_scene[tar_id0].normal();
		Point3D::VectorType tar_n2 = point3d_scene[tar_id1].normal();


		Eigen::Matrix<float, 3, 3> rotation_src_normal_to_x_axis;
		{
			Eigen::Matrix<float, 3, 3> rotationA;
			float theta = atan2(-src_n1.z(), src_n1.y());
			rotationA <<
				1, 0, 0,
				0, cos(theta), -sin(theta),
				0, sin(theta), cos(theta);

			Eigen::Matrix<float, 3, 3> rotationB;
			float alpha = atan2(-(cos(theta) * src_n1.y() - sin(theta) * src_n1.z()), src_n1.x());
			rotationB <<
				cos(alpha), -sin(alpha), 0,
				sin(alpha), cos(alpha), 0,
				0, 0, 1;
			rotation_src_normal_to_x_axis = rotationB * rotationA;
		}
		Point3D::VectorType src_p2_from_p1 = src_p2 - src_p1;
		Point3D::VectorType src_p2_from_p1_transed = rotation_src_normal_to_x_axis * src_p2_from_p1;
		float src_p2_from_p1_transed_angleXToXYPlane = atan2(-src_p2_from_p1_transed.z(), src_p2_from_p1_transed.y());

		Eigen::Matrix<float, 3, 3> rotation_tar_normal_to_x_axis;
		{
			Eigen::Matrix<float, 3, 3> rotationA;
			float theta = atan2(-tar_n1.z(), tar_n1.y());
			rotationA <<
				1, 0, 0,
				0, cos(theta), -sin(theta),
				0, sin(theta), cos(theta);

			Eigen::Matrix<float, 3, 3> rotationB;
			float alpha = atan2(-(cos(theta) * tar_n1.y() - sin(theta) * tar_n1.z()), tar_n1.x());
			rotationB <<
				cos(alpha), -sin(alpha), 0,
				sin(alpha), cos(alpha), 0,
				0, 0, 1;
			rotation_tar_normal_to_x_axis = rotationB * rotationA;
		}
		Point3D::VectorType tar_p2_from_p1 = tar_p2 - tar_p1;
		Point3D::VectorType tar_p2_from_p1_transed = rotation_tar_normal_to_x_axis * tar_p2_from_p1;
		float tar_p2_from_p1_transed_angleXToXYPlane = atan2(-tar_p2_from_p1_transed.z(), tar_p2_from_p1_transed.y());
		float angle_alpha = src_p2_from_p1_transed_angleXToXYPlane - tar_p2_from_p1_transed_angleXToXYPlane;

		Eigen::Matrix<float, 3, 3> rotationX;
		rotationX <<
			1, 0, 0,
			0, cos(angle_alpha), -sin(angle_alpha),
			0, sin(angle_alpha), cos(angle_alpha);

		Eigen::Matrix<float, 3, 3> rotation;
		rotation = rotationX * rotation_src_normal_to_x_axis;

		Point3D::VectorType tslateFromSrcToOrigin = - src_p1;
		Eigen::Matrix<float, 3, 3> rotation_tar_normal_to_x_axis_transpose =  rotation_tar_normal_to_x_axis.transpose();
		Point3D::VectorType tslateFromOriginToTar = tar_p1;

		Eigen::Matrix<float, 4, 4> trans44TranslateFromSrcToOrigin;
		trans44TranslateFromSrcToOrigin <<
			1, 0, 0, tslateFromSrcToOrigin.x(),
			0, 1, 0, tslateFromSrcToOrigin.y(),
			0, 0, 1, tslateFromSrcToOrigin.z(),
			0, 0, 0, 1;

		Eigen::Matrix<float, 4, 4> trans44Rot;
		trans44Rot.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
		trans44Rot.block<3, 3>(0, 0) = rotation;

		Eigen::Matrix<float, 4, 4> Trans44RotFromTarToXaxisTp;
		Trans44RotFromTarToXaxisTp.setIdentity();
		Trans44RotFromTarToXaxisTp.block<3, 3>(0, 0) = rotation_tar_normal_to_x_axis_transpose;

		Eigen::Matrix<float, 4, 4> trans44TslateFromOriginToTar;
		trans44TslateFromOriginToTar <<
			1, 0, 0, tslateFromOriginToTar.x(),
			0, 1, 0, tslateFromOriginToTar.y(),
			0, 0, 1, tslateFromOriginToTar.z(),
			0, 0, 0, 1;

		Eigen::Matrix<float, 4, 4> trans44;
		//trans44 = trans44TslateFromOriginToTar * Trans44RotFromTarToXaxisTp * trans44Rot * trans44TranslateFromSrcToOrigin;
		trans44 = trans44TslateFromOriginToTar * Trans44RotFromTarToXaxisTp * trans44Rot * trans44TranslateFromSrcToOrigin;

		poseEsts.at(i).trans44 = trans44;
	}
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "calculate transform " << poseEsts.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";
	/***********  verify pose  ********************/
	float best_lcp;
	int best_index;
	{
		// Build the kdtree.
		size_t number_of_points_scene = point3d_scene.size();
		
		/******* build kdtree ******/
		// Construct data structure for flann KNN
		float* pPointScene = new float[number_of_points_scene * 3];
		for (int i = 0; i < number_of_points_scene; i++)
		{
			pPointScene[i * 3 + 0] = point3d_scene[i].x();
			pPointScene[i * 3 + 1] = point3d_scene[i].y();
			pPointScene[i * 3 + 2] = point3d_scene[i].z();
		}

		start = std::chrono::high_resolution_clock::now();
		// KNN, build kd-tree
		flann::Matrix<float> dataSet(pPointScene, number_of_points_scene, 3, 3 * sizeof(float));
		flann::KDTreeCuda3dIndexParams   cudaParams(32);
		flann::KDTreeCuda3dIndex<::flann::L2<float>> KnnSearch(dataSet, cudaParams);
		KnnSearch.buildIndex();

		finish = std::chrono::high_resolution_clock::now();
		std::cout << "build KD tree for scene points " << number_of_points_scene << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";

		/******* search kdtree ******/
		int batchSize = 10000;
		size_t number_of_points_model = point3d_model.size();
		float* pPointModel = new float[number_of_points_model * 3 * batchSize];
		float* pPointModelTrans = new float[number_of_points_model * 3 * batchSize];


		int numberOfBatch = poseEsts.size() / batchSize;
		std::cout << "numberOfBatch = " << numberOfBatch << "\n";
		for (int j = 0; j < batchSize; ++j)
		{
			for (int i = 0; i < number_of_points_model; i++)
			{
				pPointModel[j * number_of_points_model + i * 3 + 0] = point3d_model[i].x();
				pPointModel[j * number_of_points_model + i * 3 + 1] = point3d_model[i].y();
				pPointModel[j * number_of_points_model + i * 3 + 2] = point3d_model[i].z();
			}
		}

		flann::Matrix<int> indices(new int[batchSize * number_of_points_model * 1], batchSize* number_of_points_model, 1);
		flann::Matrix<float> dists(new float[batchSize * number_of_points_model * 1], batchSize* number_of_points_model, 1);
		flann::SearchParams searchParam(8, 0, true);

		start = std::chrono::high_resolution_clock::now();

		for (int k = 0; k < numberOfBatch; ++k)
		{	
			// transform
			for (int j = 0; j < batchSize; ++j)
			{
				for (int i = 0; i < number_of_points_model; i++)
				{
					Eigen::Vector3f v = (poseEsts[k * batchSize + j].trans44 * point3d_model[i].pos().homogeneous()).head<3>();
					pPointModelTrans[j * number_of_points_model + i * 3 + 0] = v.x();
					pPointModelTrans[j * number_of_points_model + i * 3 + 1] = v.y();
					pPointModelTrans[j * number_of_points_model + i * 3 + 2] = v.z();
				}
			}
			// KNN, search neighbor
			flann::Matrix<float> modelSet(pPointModelTrans, batchSize * number_of_points_model, 3, 3 * sizeof(float));
			KnnSearch.knnSearchGpu(modelSet, indices, dists, 1, searchParam);

		}

		delete[] pPointScene;
		delete[] pPointModel;
		delete[] pPointModelTrans;
		delete[] indices.ptr();
		delete[] dists.ptr();

		best_index = 0;
		finish = std::chrono::high_resolution_clock::now();
		std::cout << "verify transform " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";
	}

	/***********  show best pose  ********************/
	{
		std::vector<Point3D> point3d_model_pose;

		point3d_model_pose.clear();
		rgbd::transform_pointset(point3d_model, point3d_model_pose, poseEsts[best_index].trans44);
		rgbd::save_as_ply(debug_path + "/best_pose.ply", point3d_model_pose, 1);
		rgbd::save_as_ply(debug_path + "/scene.ply", point3d_scene, 1);
	}

	return 0;
}