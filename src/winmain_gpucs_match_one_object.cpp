#include <filesystem>
#include <algorithm>
#include "rgbd.hpp"
#include "accelerators/kdtree.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include "model_match.cuh"

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
static int batchSize = 10000;
static int debug_flag = 0;

//#define singleTest
int gpucs(std::string scene_path, std::string object_name)
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

#define CPU_Trans_cal (0)
#ifdef CPU_Trans_cal
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
#else 
	/***********  calculate transform for each ppf correspondences ********************/
	{
		size_t number_of_points_model = point3d_model.size();
		float* pPointModel = new float[number_of_points_model * 3];
		for (int i = 0; i < number_of_points_model; i++)
		{
			pPointModel[i * 3 + 0] = point3d_model[i].x();
			pPointModel[i * 3 + 1] = point3d_model[i].y();
			pPointModel[i * 3 + 2] = point3d_model[i].z();
		}
		float* pPointModelNormal = new float[number_of_points_model * 3];
		for (int i = 0; i < number_of_points_model; i++)
		{
			pPointModelNormal[i * 3 + 0] = point3d_model[i].normal().x();
			pPointModelNormal[i * 3 + 1] = point3d_model[i].normal().y();
			pPointModelNormal[i * 3 + 2] = point3d_model[i].normal().z();
		}

		size_t number_of_points_scene = point3d_scene.size();
		float* pPointScene = new float[number_of_points_scene * 3];
		for (int i = 0; i < number_of_points_scene; i++)
		{
			pPointScene[i * 3 + 0] = point3d_scene[i].x();
			pPointScene[i * 3 + 1] = point3d_scene[i].y();
			pPointScene[i * 3 + 2] = point3d_scene[i].z();
		}
		float* pPointSceneNormal = new float[number_of_points_scene * 3];
		for (int i = 0; i < number_of_points_scene; i++)
		{
			pPointSceneNormal[i * 3 + 0] = point3d_scene[i].normal().x();
			pPointSceneNormal[i * 3 + 1] = point3d_scene[i].normal().y();
			pPointSceneNormal[i * 3 + 2] = point3d_scene[i].normal().z();
		}
		float* d_pPointModelGPU;
		float* d_pPointSceneGPU;
		cudaMalloc((void**)&d_pPointModelGPU, sizeof(float)* number_of_points_model * 3);
		cudaMalloc((void**)&d_pPointSceneGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointModelGPU, pPointModel, sizeof(float)* number_of_points_model * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pPointSceneGPU, pPointScene, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);

		float* d_pPointModelNormalGPU;
		float* d_pPointSceneNormalGPU;
		cudaMalloc((void**)&d_pPointModelNormalGPU, sizeof(float)* number_of_points_model * 3);
		cudaMalloc((void**)&d_pPointSceneNormalGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointModelNormalGPU, pPointModelNormal, sizeof(float)* number_of_points_model * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pPointSceneNormalGPU, pPointSceneNormal, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);

		float* pAllPoses = new float[16 * poseEsts.size()];
		memset(pAllPoses, 0, sizeof(float) * 16 * poseEsts.size());

		float* d_pPosesGPU_Batch;
		cudaMalloc((void**)&d_pPosesGPU_Batch, sizeof(float)* batchSize * 16);

		int* pSrcId0_Batch = new int[batchSize];
		int* pSrcId1_Batch = new int[batchSize];
		int* pTarId0_Batch = new int[batchSize];
		int* pTarId1_Batch = new int[batchSize];

		int* d_pSrcId0_GPUBatch;
		int* d_pSrcId1_GPUBatch;
		int* d_pTarId0_GPUBatch;
		int* d_pTarId1_GPUBatch;
		cudaMalloc((void**)&d_pSrcId0_GPUBatch, sizeof(int)* batchSize);
		cudaMalloc((void**)&d_pSrcId1_GPUBatch, sizeof(int)* batchSize);
		cudaMalloc((void**)&d_pTarId0_GPUBatch, sizeof(int)* batchSize);
		cudaMalloc((void**)&d_pTarId1_GPUBatch, sizeof(int)* batchSize);

		start = std::chrono::high_resolution_clock::now();

		int numOfBatch = poseEsts.size() / batchSize;
		for (int k = 0; k < numOfBatch; ++k)
		{
			for (int i = 0; i < batchSize; ++i)
			{
				pSrcId0_Batch[i] = poseEsts.at(k * batchSize + i).srcId[0];
				pSrcId1_Batch[i] = poseEsts.at(k * batchSize + i).srcId[1];
				pTarId0_Batch[i] = poseEsts.at(k * batchSize + i).tarId[0];
				pTarId1_Batch[i] = poseEsts.at(k * batchSize + i).tarId[1];
			}
			cudaMemcpy(d_pSrcId0_GPUBatch, pSrcId0_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_pSrcId1_GPUBatch, pSrcId1_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_pTarId0_GPUBatch, pTarId0_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_pTarId1_GPUBatch, pTarId1_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);

			ComputeTransformForCorrespondencesCU(
				d_pPointModelGPU,
				d_pPointModelNormalGPU,
				number_of_points_model,
				d_pPointSceneGPU,	
				d_pPointSceneNormalGPU,
				number_of_points_scene,
				d_pSrcId0_GPUBatch,
				d_pSrcId1_GPUBatch,
				d_pTarId0_GPUBatch,
				d_pTarId1_GPUBatch,
				batchSize,
				d_pPosesGPU_Batch				
			);
			cudaMemcpy(pAllPoses + (k * batchSize * 16), d_pPosesGPU_Batch, sizeof(float) * batchSize * 16, cudaMemcpyDeviceToHost);
		}


		finish = std::chrono::high_resolution_clock::now();
		std::cout << "calculate transform " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";

		for (int i = 0; i < numOfBatch * batchSize; ++i)
		{
			Eigen::Matrix<float, 4, 4> trans44;

			trans44 <<
				pAllPoses[i * 16 + 0], pAllPoses[i * 16 + 1], pAllPoses[i * 16 + 2], pAllPoses[i * 16 + 3],
				pAllPoses[i * 16 + 4], pAllPoses[i * 16 + 5], pAllPoses[i * 16 + 6], pAllPoses[i * 16 + 7],
				pAllPoses[i * 16 + 8], pAllPoses[i * 16 + 9], pAllPoses[i * 16 + 10], pAllPoses[i * 16 + 11],
				pAllPoses[i * 16 + 12], pAllPoses[i * 16 + 13], pAllPoses[i * 16 + 14], pAllPoses[i * 16 + 15];

			poseEsts.at(i).trans44 = trans44;
		}

		delete[] pPointModel;
		delete[] pPointModelNormal;
		delete[] pPointScene;
		delete[] pPointSceneNormal;
		cudaFree(d_pPointModelGPU);
		cudaFree(d_pPointSceneGPU);
		cudaFree(d_pPointModelNormalGPU);
		cudaFree(d_pPointSceneNormalGPU);
		delete[] pAllPoses;
		cudaFree(d_pPosesGPU_Batch);

		delete[] pSrcId0_Batch;
		delete[] pSrcId1_Batch;
		delete[] pTarId0_Batch;
		delete[] pTarId1_Batch;
		cudaFree(d_pSrcId0_GPUBatch);
		cudaFree(d_pSrcId1_GPUBatch);
		cudaFree(d_pTarId0_GPUBatch);
		cudaFree(d_pTarId1_GPUBatch);
	}
#endif

	if (1)
	{
		std::string path;
		path = debug_path + "/transformations.txt";
		std::FILE* f = std::fopen(path.c_str(), "w");

		for (int i = 0; i < poseEsts.size(); ++i)
		{
			std::fprintf(f, "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
				poseEsts[i].trans44(0, 0), poseEsts[i].trans44(0, 1), poseEsts[i].trans44(0, 2), poseEsts[i].trans44(0, 3),
				poseEsts[i].trans44(1, 0), poseEsts[i].trans44(1, 1), poseEsts[i].trans44(1, 2), poseEsts[i].trans44(1, 3),
				poseEsts[i].trans44(2, 0), poseEsts[i].trans44(2, 1), poseEsts[i].trans44(2, 2), poseEsts[i].trans44(2, 3),
				poseEsts[i].trans44(3, 0), poseEsts[i].trans44(3, 1), poseEsts[i].trans44(3, 2), poseEsts[i].trans44(3, 3)
			);
		}
		std::fclose(f);
	}

#ifdef singleTest
	//std::vector<poseEst> fakeposes;
	//fakeposes.push_back(poseEsts.at(324378));
	//poseEsts = fakeposes;
#endif
	/***********  verify pose  ********************/
	float best_lcp = 0;
	int best_index = 0;
	{
		/**********
		prebuilt 
		************/
		size_t number_of_points_model = point3d_model.size();
		float* pPointModel = new float[number_of_points_model * 3 * batchSize];
		
		float* d_pPointModelGPU;
		cudaMalloc((void**)&d_pPointModelGPU, sizeof(float)* number_of_points_model * 3 * batchSize);
		float* d_pPointModelTransGPU;
		cudaMalloc((void**)&d_pPointModelTransGPU, sizeof(float)* number_of_points_model * 3 * batchSize);

		float* pPoses = new float[16 * batchSize];

		float* d_pPosesGPU;
		cudaMalloc((void**)&d_pPosesGPU, sizeof(float)* batchSize * 16);

		int* d_ppointsValidsGPU;
		cudaMalloc((void**)&d_ppointsValidsGPU, sizeof(int)* number_of_points_model * batchSize);

		float* d_pLCPsGPU;
		cudaMalloc((void**)&d_pLCPsGPU, sizeof(float)* batchSize);

		/********** assign points ***********/
		for (int j = 0; j < batchSize; ++j)
		{
			for (int i = 0; i < number_of_points_model; i++)
			{
				pPointModel[j * number_of_points_model * 3 + i * 3 + 0] = point3d_model[i].x();
				pPointModel[j * number_of_points_model * 3 + i * 3 + 1] = point3d_model[i].y();
				pPointModel[j * number_of_points_model * 3 + i * 3 + 2] = point3d_model[i].z();
			}
		}
		cudaMemcpy(d_pPointModelGPU, pPointModel, sizeof(float)* number_of_points_model * batchSize * 3, cudaMemcpyHostToDevice);

		int* d_indices_gpu;
		cudaMalloc((void**)&d_indices_gpu, sizeof(int) * batchSize * number_of_points_model);
		float* d_dists_gpu;
		cudaMalloc((void**)&d_dists_gpu, sizeof(float) * batchSize* number_of_points_model);

		flann::Matrix<int> indices_gpu(d_indices_gpu, batchSize * number_of_points_model, 1, 1 * sizeof(int));
		flann::Matrix<float> dists_gpu(d_dists_gpu, batchSize * number_of_points_model, 1, 1 * sizeof(float));
		flann::SearchParams searchParam_gpu;
		searchParam_gpu.matrices_in_gpu_ram = true;

		/**********
		runtime built
		************/
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

		/*
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
		*/
		float* d_pPointSceneGPU;
		cudaMalloc((void**)&d_pPointSceneGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointSceneGPU, pPointScene, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);


		int numberOfBatch = poseEsts.size() / batchSize;
#ifdef singleTest
		numberOfBatch = 1;
#endif
		std::cout << "numberOfBatch = " << numberOfBatch << "\n";

		float* pLCPs = new float[numberOfBatch * batchSize];
		memset(pLCPs, 0, sizeof(float) * numberOfBatch * batchSize);

		/**********
		runtime
		************/
		/********** search kdtree ************/
		//for (int mm = 0; mm < 10000; ++mm)
		{
			//getchar();
			start = std::chrono::high_resolution_clock::now();
			for (int k = 0; k < numberOfBatch; ++k)
			{
				
				// assign pose to GPU
				for (int j = 0; j < batchSize; ++j)
				{
					for (int i = 0; i < 16; i++)
					{
						int row = i / 4;
						int col = i % 4;
						pPoses[j * 16 + i] = poseEsts[k * batchSize + j].trans44(row, col);
					}
				}
				cudaMemcpy(d_pPosesGPU, pPoses, sizeof(float) * 16 * batchSize, cudaMemcpyHostToDevice);
				
				TransformPointsCU(
					d_pPointModelGPU,
					number_of_points_model,
					batchSize,
					d_pPosesGPU,
					d_pPointModelTransGPU
				);

				/*
				 if (debug_flag == 1)
				 {
					 std::string path;
					 path = debug_path + "/model.obj";
					 std::FILE* f = std::fopen(path.c_str(), "w");

					 for (int i = 0; i < number_of_points_model; ++i)
					 {
						 std::fprintf(f, "v %f %f %f \n", pPointModel[i * 3 + 0], pPointModel[i * 3 + 1], pPointModel[i * 3 + 2]);
					 }
					 std::fclose(f);
				 }
				 if (debug_flag == 1)
				 {
					 float* pPointModelTransGPU_DEBUG = new float[number_of_points_model * 3 * batchSize];
					 cudaMemcpy(pPointModelTransGPU_DEBUG, d_pPointModelTransGPU, sizeof(float)* number_of_points_model * 3 * batchSize, cudaMemcpyDeviceToHost);

					 std::string path;
					 path = debug_path + "/trans.obj";
					 std::FILE* f = std::fopen(path.c_str(), "w");

					 for (int i = 0; i < number_of_points_model * batchSize; ++i)
					 {
						 std::fprintf(f, "v %f %f %f \n", pPointModelTransGPU_DEBUG[i * 3 + 0], pPointModelTransGPU_DEBUG[i * 3 + 1], pPointModelTransGPU_DEBUG[i * 3 + 2]);
					 }
					 std::fclose(f);

					 delete [] pPointModelTransGPU_DEBUG;

				 }
				 */
				 /*
				 // TransformPoints Normally for debug
				 if (debug_flag == 1)
				 {
					 std::string path;
					 path = debug_path + "/transDebug.obj";
					 std::FILE* f = std::fopen(path.c_str(), "w");
					 for (int i = 0; i < number_of_points_model; ++i)
					 {
						 Point3D::VectorType tp = (poseEsts[k * batchSize + 0].trans44 * point3d_model[i].pos().homogeneous()).head<3>();
						 std::fprintf(f, "v %f %f %f \n", tp[0], tp[1], tp[2]);
					 }
					 std::fclose(f);
				 }

				 if (debug_flag == 1)
				 {
					 std::string path;
					 path = debug_path + "/SceneD.obj";
					 std::FILE* f = std::fopen(path.c_str(), "w");
					 for (int i = 0; i < number_of_points_scene; ++i)
					 {
						 std::fprintf(f, "v %f %f %f \n", pPointScene[i * 3 + 0], pPointScene[i * 3 + 1], pPointScene[i * 3 + 2]);
					 }
					 std::fclose(f);
				 }
				 */
				 /*
				// KNN, search neighbor
				 if (debug_flag == 1)
				 {
					 float* pPointScene_2 = new float[number_of_points_scene * 3];
					 for (int i = 0; i < number_of_points_scene; ++i)
					 {
						 pPointScene_2[i * 3 + 0] = pPointScene[i * 3 + 0] + 0.001;
						 pPointScene_2[i * 3 + 1] = pPointScene[i * 3 + 1] + 0.001;
						 pPointScene_2[i * 3 + 2] = pPointScene[i * 3 + 2] + 0.001;
					 }

					 flann::Matrix<float> dataSet(pPointScene_2, number_of_points_scene, 3, 3 * sizeof(float));
					 flann::Matrix<int> indices(new int[dataSet.rows], dataSet.rows, 1);
					 flann::Matrix<float> dists(new float[dataSet.rows], dataSet.rows, 1);
					 flann::SearchParams searchParam(8, 0, true);
					 KnnSearch.knnSearchGpu(dataSet, indices, dists, 1, searchParam);
					 {
						 std::string path;
						 path = debug_path + "/indices_scene_gput.txt";
						 std::FILE* f = std::fopen(path.c_str(), "w");

						 for (int i = 0; i < number_of_points_scene; ++i)
						 {
							 std::fprintf(f, "%d\n", indices.ptr()[i]);
						 }
						 std::fclose(f);
					 }
					 {
						 std::string path;
						 path = debug_path + "/dists_scene_gput.txt";
						 std::FILE* f = std::fopen(path.c_str(), "w");

						 for (int i = 0; i < number_of_points_scene; ++i)
						 {
							 std::fprintf(f, "%f\n", dists.ptr()[i]);
						 }
						 std::fclose(f);
					 }
					 delete[] indices.ptr();
					 delete[] dists.ptr();
					 delete[] pPointScene_2;
				 }

				 if (debug_flag == 1)
				 {
					 float* pPointModelTransGPU_2 = new float[number_of_points_model * 3 * batchSize];
					 cudaMemcpy(pPointModelTransGPU_2, d_pPointModelTransGPU, sizeof(float)* number_of_points_model * 3 * batchSize, cudaMemcpyDeviceToHost);

					 flann::Matrix<float> dataSet(pPointModelTransGPU_2, number_of_points_model  * batchSize, 3, 3 * sizeof(float));
					 flann::Matrix<int> indices(new int[dataSet.rows], dataSet.rows, 1);
					 flann::Matrix<float> dists(new float[dataSet.rows], dataSet.rows, 1);
					 flann::SearchParams searchParam(8, 0, true);
					 KnnSearch.knnSearchGpu(dataSet, indices, dists, 1, searchParam);
					 {
						 std::string path;
						 path = debug_path + "/indices_gput.txt";
						 std::FILE* f = std::fopen(path.c_str(), "w");

						 for (int i = 0; i < number_of_points_model * batchSize; ++i)
						 {
							 std::fprintf(f, "%d\n", indices.ptr()[i]);
						 }
						 std::fclose(f);
					 }
					 {
						 std::string path;
						 path = debug_path + "/dists_gput.txt";
						 std::FILE* f = std::fopen(path.c_str(), "w");

						 for (int i = 0; i < number_of_points_model * batchSize; ++i)
						 {
							 std::fprintf(f, "%f\n", dists.ptr()[i]);
						 }
						 std::fclose(f);
					 }
					 delete[] indices.ptr();
					 delete[] dists.ptr();
				 }
				 */

				/*
				flann::Matrix<float> modelSet_gpu(d_pPointModelTransGPU, batchSize* number_of_points_model, 3, 3 * sizeof(float));
				KnnSearch.knnSearchGpu(modelSet_gpu, indices_gpu, dists_gpu, 1, searchParam_gpu);
				if (debug_flag == 1)
				{
					{
						int* pindices_gpu_DEBUG = new int[number_of_points_model * batchSize];
						cudaMemcpy(pindices_gpu_DEBUG, d_indices_gpu, sizeof(int)* batchSize * number_of_points_model, cudaMemcpyDeviceToHost);
						std::string path;
						path = debug_path + "/indices.txt";
						std::FILE* f = std::fopen(path.c_str(), "w");

						for (int i = 0; i < number_of_points_model * batchSize; ++i)
						{
							std::fprintf(f, "%d\n", pindices_gpu_DEBUG[i]);
						}
						std::fclose(f);
					}
					{
						float* pdists_gpu_DEBUG = new float[number_of_points_model * batchSize];
						cudaMemcpy(pdists_gpu_DEBUG, d_dists_gpu, sizeof(float)* batchSize * number_of_points_model, cudaMemcpyDeviceToHost);

						std::string path;
						path = debug_path + "/dists.txt";
						std::FILE* f = std::fopen(path.c_str(), "w");

						for (int i = 0; i < number_of_points_model * batchSize; ++i)
						{
							std::fprintf(f, "%f\n", pdists_gpu_DEBUG[i]);
						}
						std::fclose(f);
					}
				}

				
				float sqdistThd = distance_threshold * distance_threshold; // since flann dist is sqdist
				verifyPointsNearCU(
					d_dists_gpu,
					number_of_points_model,
					batchSize,
					sqdistThd,
					d_ppointsValidsGPU);
				*/
				
				float sqdistThd = distance_threshold * distance_threshold; // since flann dist is sqdist
				findAndVerifyNearestPointsCU(
					d_pPointModelTransGPU,
					number_of_points_model,
					d_pPointSceneGPU,
					number_of_points_scene,
					batchSize,
					sqdistThd,
					d_ppointsValidsGPU);

				if (debug_flag == 1)
				{
					int* ppointsValidsGPU_DEBUG = new int[number_of_points_model * batchSize];
					cudaMemcpy(ppointsValidsGPU_DEBUG, d_ppointsValidsGPU, sizeof(int)* number_of_points_model * batchSize, cudaMemcpyDeviceToHost);

					std::string path;
					path = debug_path + "/isValid.txt";
					std::FILE* f = std::fopen(path.c_str(), "w");

					for (int i = 0; i < number_of_points_model; ++i)
					{
						std::fprintf(f, "%d\n", ppointsValidsGPU_DEBUG[i]);
					}
					std::fclose(f);
				}
				
				computePoseLCP_CU(
					d_ppointsValidsGPU,
					number_of_points_model,
					batchSize,
					d_pLCPsGPU);

				cudaMemcpy(pLCPs + (k * batchSize), d_pLCPsGPU, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);
			}

			float* maxAddr = std::max_element(pLCPs, pLCPs + (numberOfBatch * batchSize));
			int maxLCPIdx = std::distance(pLCPs, maxAddr);
			std::cout << "max LCP at: " << maxLCPIdx << " , LCP = " << pLCPs[maxLCPIdx] << '\n';

			best_index = maxLCPIdx;

			finish = std::chrono::high_resolution_clock::now();
			std::cout << "verify transform " << poseEsts.size() << " in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
				<< " milliseconds\n";
		}

		if (1)
		{
			std::string path;
			path = debug_path + "/LCP.txt";
			std::FILE* f = std::fopen(path.c_str(), "w");

			for (int i = 0; i < numberOfBatch * batchSize; ++i)
			{
				std::fprintf(f, "%f\n", pLCPs[i]);
			}
			std::fclose(f);
		}

		delete[] pPointScene;
		/**********
		final free
		************/
		delete[] pPoses;
		delete[] pPointModel;
		delete[] pLCPs;
		cudaFree(d_pPointModelGPU);
		cudaFree(d_pPointModelTransGPU);
		cudaFree(d_indices_gpu);
		cudaFree(d_dists_gpu);
		cudaFree(d_pPosesGPU);
		cudaFree(d_pLCPsGPU);
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