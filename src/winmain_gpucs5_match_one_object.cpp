#include <filesystem>
#include <algorithm>
#include "rgbd.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#define FLANN_USE_CUDA
#include <flann/flann.hpp>

#include "model_match.cuh"

namespace fs = std::experimental::filesystem;

// alg parameters
static float ds_voxel_size = 0.005; // In m, sampling size
static float normal_radius = 0.005; // In m radius for calculating normal vector of a point, depends on point cloud density
static int ppf_tr_discretization = 5; // In mm, same as when create model. 
static int ppf_rot_discretization = 5; // degrees, same as when create model. 
static float validPairMinDist = 0.03; // depends on object size

static float lcp_distthreshold = 0.002; // for Congruent Set Matching and LCP computation, depends on point cloud density

// input parameters
static std::string scene_scale = "mm";		// input model scale

// running parameters
static int batchSize = 10000;
static int debug_flag = 0;

int gpucs5(std::string scene_path, std::string object_path, std::string ppf_path)
{
	std::string x_location = scene_path + "/XMap.tif";
	std::string y_location = scene_path + "/YMap.tif";
	std::string z_location = scene_path + "/ZMap.tif";
	std::string isValid_location = scene_path + "/isValid.bmp";
	std::string segmentation_map_location = scene_path + "/segmentation.bmp";
	std::string model_map_path = ppf_path;
	std::string model_location = object_path;
	std::string debug_path = "./dbg";
	/***********  debug path ********************/
	fs::create_directories(debug_path);

	/***********  load PPF map ********************/
	PPFMapType model_map;
	rgbd::load_ppf_map(model_map_path, model_map);

	/********** load model ********************************/
	std::vector<Point3D> point3d_model;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::io::loadPLYFile(model_location, *cloud);
	rgbd::load_ply_model(cloud, point3d_model, 1.0f);

	rgbd::save_as_ply(debug_path + "/model.ply", point3d_model, 1.0);
	std::cout << "|M| = " << point3d_model.size() << ",  |map(M)| = " << model_map.size() << std::endl;

	/***********  load scene in sample form ********************/
	float inscale = 1.0;
	if (scene_scale == "m")
	{
		inscale = 1.0;
	}
	else if (scene_scale == "mm")
	{
		inscale = 1.0 / 1000.0;
	}

	std::vector<Point3D> point3d_scene;
	rgbd::load_xyzmap_data_sampled(
		x_location,
		y_location,
		z_location,
		inscale,
		isValid_location,
		segmentation_map_location,
		ds_voxel_size,
		normal_radius,
		point3d_scene);

	/***********  move to origin********************/
	Point3D minPt;
	Point3D maxPt;
	minPt = point3d_scene[0];
	maxPt = point3d_scene[0];

	for (int i = 0; i < point3d_scene.size(); ++i)
	{
		if (minPt.x() > point3d_scene[i].x())
		{
			minPt.x() = point3d_scene[i].x();
		}
		if (minPt.y() > point3d_scene[i].y())
		{
			minPt.y() = point3d_scene[i].y();
		}
		if (minPt.z() > point3d_scene[i].z())
		{
			minPt.z() = point3d_scene[i].z();
		}
	}
	for (int i = 0; i < point3d_scene.size(); ++i)
	{
		if (maxPt.x() < point3d_scene[i].x())
		{
			maxPt.x() = point3d_scene[i].x();
		}
		if (maxPt.y() < point3d_scene[i].y())
		{
			maxPt.y() = point3d_scene[i].y();
		}
		if (maxPt.z() < point3d_scene[i].z())
		{
			maxPt.z() = point3d_scene[i].z();
		}
	}
	float appcentroid[3];
	appcentroid[0] = (minPt.x() + maxPt.x()) * 0.5;
	appcentroid[1] = (minPt.y() + maxPt.y()) * 0.5;
	appcentroid[2] = (minPt.z() + maxPt.z()) * 0.5;

	for (int i = 0; i < point3d_scene.size(); ++i)
	{
		point3d_scene[i].x() = point3d_scene[i].x() - appcentroid[0];
		point3d_scene[i].y() = point3d_scene[i].y() - appcentroid[1];
		point3d_scene[i].z() = point3d_scene[i].z() - appcentroid[2];
	}

	float centroid[3];
	centroid[0] = 0;
	centroid[1] = 0;
	centroid[2] = 0;

	for (int i = 0; i < point3d_scene.size(); ++i)
	{
		centroid[0] += point3d_scene[i].x();
		centroid[1] += point3d_scene[i].y();
		centroid[2] += point3d_scene[i].z();
	}

	centroid[0] = centroid[0] / (float)point3d_scene.size();
	centroid[1] = centroid[1] / (float)point3d_scene.size();
	centroid[2] = centroid[2] / (float)point3d_scene.size();

	for (int i = 0; i < point3d_scene.size(); ++i)
	{
		point3d_scene[i].x() = point3d_scene[i].x() - centroid[0];
		point3d_scene[i].y() = point3d_scene[i].y() - centroid[1];
		point3d_scene[i].z() = point3d_scene[i].z() - centroid[2];
	}

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
				continue;
			}
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
				e.srcId[0] = v[k].first;  // model
				e.srcId[1] = v[k].second;
				e.tarId[0] = firstIdx; // scene
				e.tarId[1] = secondIdx;
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
	
	{
		int numOfBatch = poseEsts.size() / batchSize + 1;
		int totalPoseSize = poseEsts.size();
		int totalComputeSize = numOfBatch * batchSize;
		std::cout << "numberOfBatch = " << numOfBatch << "\n";
		std::cout << "BatchSize = " << batchSize << "\n";
		std::cout << "totalPoseSize = " << totalPoseSize << "\n";
		std::cout << "totalComputeSize = " << totalComputeSize << "\n";

		float* pAllPoses = new float[16 * totalComputeSize];
		memset(pAllPoses, 0, sizeof(float) * 16 * totalComputeSize);

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

		for (int k = 0; k < numOfBatch; ++k)
		{
			for (int i = 0; i < batchSize; ++i)
			{
				int currentPoseIdx = k * batchSize + i;
				if (currentPoseIdx < totalPoseSize)
				{
					pSrcId0_Batch[i] = poseEsts.at(k * batchSize + i).srcId[0];
					pSrcId1_Batch[i] = poseEsts.at(k * batchSize + i).srcId[1];
					pTarId0_Batch[i] = poseEsts.at(k * batchSize + i).tarId[0];
					pTarId1_Batch[i] = poseEsts.at(k * batchSize + i).tarId[1];
				}

				if (currentPoseIdx == 384786 || currentPoseIdx == 384799)
				{
					std::cout << i << " " << pSrcId0_Batch[i] << " " << pSrcId1_Batch[i] << " " << pTarId0_Batch[i] << " " << pTarId1_Batch[i] << "\n";
				}
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

		for (int i = 0; i < totalPoseSize; ++i)
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

	if (0)
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

	/***********  verify pose  ********************/
	float best_lcp = 0;
	int best_index = 0;
	{
		/**********
		prebuilt 
		************/
		size_t number_of_points_model = point3d_model.size();
		float* pPointModelBatch = new float[number_of_points_model * 3 * batchSize];
		
		float* d_pPointModelGPUBatch;
		cudaMalloc((void**)&d_pPointModelGPUBatch, sizeof(float)* number_of_points_model * 3 * batchSize);
		float* d_pPointModelTransGPUBatch;
		cudaMalloc((void**)&d_pPointModelTransGPUBatch, sizeof(float)* number_of_points_model * 3 * batchSize);

		float* pPosesBatch = new float[16 * batchSize];

		float* d_pPosesGPUBatch;
		cudaMalloc((void**)&d_pPosesGPUBatch, sizeof(float)* batchSize * 16);

		int* d_ppointsValidsGPUBatch;
		cudaMalloc((void**)&d_ppointsValidsGPUBatch, sizeof(int)* number_of_points_model * batchSize);

		float* d_pLCPsGPUBatch;
		cudaMalloc((void**)&d_pLCPsGPUBatch, sizeof(float)* batchSize);

		/********** assign points ***********/
		for (int j = 0; j < batchSize; ++j)
		{
			for (int i = 0; i < number_of_points_model; i++)
			{
				pPointModelBatch[j * number_of_points_model * 3 + i * 3 + 0] = point3d_model[i].x();
				pPointModelBatch[j * number_of_points_model * 3 + i * 3 + 1] = point3d_model[i].y();
				pPointModelBatch[j * number_of_points_model * 3 + i * 3 + 2] = point3d_model[i].z();
			}
		}
		cudaMemcpy(d_pPointModelGPUBatch, pPointModelBatch, sizeof(float)* number_of_points_model * batchSize * 3, cudaMemcpyHostToDevice);

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
		size_t number_of_points_scene = point3d_scene.size();
		float* pPointScene = new float[number_of_points_scene * 3];
		for (int i = 0; i < number_of_points_scene; i++)
		{
			pPointScene[i * 3 + 0] = point3d_scene[i].x();
			pPointScene[i * 3 + 1] = point3d_scene[i].y();
			pPointScene[i * 3 + 2] = point3d_scene[i].z();
		}

		float* d_pPointSceneGPU;
		cudaMalloc((void**)&d_pPointSceneGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointSceneGPU, pPointScene, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);


		int numberOfBatch = poseEsts.size() / batchSize + 1;
		int totalPoseNum = poseEsts.size();
		int totalComputeNum = numberOfBatch * batchSize;
		float* pLCPs = new float[totalComputeNum];
		memset(pLCPs, 0, sizeof(float) * totalComputeNum);

		// KNN, build kd-tree
		start = std::chrono::high_resolution_clock::now();
		flann::Matrix<float> dataSet(pPointScene, number_of_points_scene, 3, 3 * sizeof(float));
		flann::KDTreeCuda3dIndexParams   cudaParams(32);
		flann::KDTreeCuda3dIndex<::flann::L2<float>> KnnSearch(dataSet, cudaParams);
		KnnSearch.buildIndex();

		finish = std::chrono::high_resolution_clock::now();
		std::cout << "build KD tree for scene points " << number_of_points_scene << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";

		/**********
		runtime
		************/
		start = std::chrono::high_resolution_clock::now();
		for (int k = 0; k < numberOfBatch; ++k)
		{
			// assign pose to GPU
			for (int j = 0; j < batchSize; ++j)
			{
				int currentPoseIdx = k * batchSize + j;
				if (currentPoseIdx < totalPoseNum)
				{
					for (int i = 0; i < 16; i++)
					{
						int row = i / 4;
						int col = i % 4;
						pPosesBatch[j * 16 + i] = poseEsts[k * batchSize + j].trans44(row, col);
					}
				}
			}
			cudaMemcpy(d_pPosesGPUBatch, pPosesBatch, sizeof(float) * 16 * batchSize, cudaMemcpyHostToDevice);

			TransformPointsCU(
				d_pPointModelGPUBatch,
				number_of_points_model,
				batchSize,
				d_pPosesGPUBatch,
				d_pPointModelTransGPUBatch
			);

			flann::Matrix<float> modelSet_gpu(d_pPointModelTransGPUBatch, batchSize* number_of_points_model, 3, 3 * sizeof(float));
			KnnSearch.knnSearchGpu(modelSet_gpu, indices_gpu, dists_gpu, 1, searchParam_gpu);

			float sqdistThd = lcp_distthreshold * lcp_distthreshold; // since flann dist is sqdist
			verifyPointsNearCU(
				d_dists_gpu,
				number_of_points_model,
				batchSize,
				sqdistThd,
				d_ppointsValidsGPUBatch);
			
			computePoseLCP_CU(
				d_ppointsValidsGPUBatch,
				number_of_points_model,
				batchSize,
				d_pLCPsGPUBatch);

			cudaMemcpy(pLCPs + (k * batchSize), d_pLCPsGPUBatch, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);
		}

		float* maxAddr = std::max_element(pLCPs, pLCPs + totalPoseNum);
		int maxLCPIdx = std::distance(pLCPs, maxAddr);
		std::cout << "max LCP at: " << maxLCPIdx << " , LCP = " << pLCPs[maxLCPIdx] << '\n';

		best_index = maxLCPIdx;

		finish = std::chrono::high_resolution_clock::now();
		std::cout << "verify transform " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";

		if (1)
		{
			std::string path;
			path = debug_path + "/LCP.txt";
			std::FILE* f = std::fopen(path.c_str(), "w");

			for (int i = 0; i < totalPoseNum; ++i)
			{
				std::fprintf(f, "%f\n", pLCPs[i]);
			}
			std::fclose(f);
		}

		delete[] pPointScene;
		/**********
		final free
		************/
		delete[] pPosesBatch;
		delete[] pPointModelBatch;
		delete[] pLCPs;
		cudaFree(d_pPointModelGPUBatch);
		cudaFree(d_pPointModelTransGPUBatch);
		cudaFree(d_pPosesGPUBatch);
		cudaFree(d_pLCPsGPUBatch);
	}

	/***********  show best pose  ********************/
	{
		std::vector<Point3D> point3d_model_pose;

		point3d_model_pose.clear();
		//rgbd::transform_pointset(point3d_model, point3d_model_pose, poseEsts[best_index].trans44);

		std::cout << "best pose trans44 = \n";
		char buffer[1024];
		snprintf(buffer, 1024, "%.3f %.3f %.3f %.3f\n",
			poseEsts[best_index].trans44(0, 0), poseEsts[best_index].trans44(0, 1), poseEsts[best_index].trans44(0, 2), poseEsts[best_index].trans44(0, 3));
		std::cout << buffer;
		snprintf(buffer, 1024, "%.3f %.3f %.3f %.3f\n",
			poseEsts[best_index].trans44(1, 0), poseEsts[best_index].trans44(1, 1), poseEsts[best_index].trans44(1, 2), poseEsts[best_index].trans44(1, 3));
		std::cout << buffer;
		snprintf(buffer, 1024, "%.3f %.3f %.3f %.3f\n",
			poseEsts[best_index].trans44(2, 0), poseEsts[best_index].trans44(2, 1), poseEsts[best_index].trans44(2, 2), poseEsts[best_index].trans44(2, 3));
		std::cout << buffer;
		snprintf(buffer, 1024, "%.3f %.3f %.3f %.3f\n",
			poseEsts[best_index].trans44(3, 0), poseEsts[best_index].trans44(3, 1), poseEsts[best_index].trans44(3, 2), poseEsts[best_index].trans44(3, 3));
		std::cout << buffer;

		rgbd::transform_pointset(point3d_model, point3d_model_pose, poseEsts[best_index].trans44);
		rgbd::save_as_ply(debug_path + "/best_pose.ply", point3d_model_pose, 1);
		rgbd::save_as_ply(debug_path + "/scene.ply", point3d_scene, 1);

		if (scene_scale == "mm")
		{
			rgbd::save_as_ply(debug_path + "/best_pose_1000.ply", point3d_model_pose, 1000);
		}
		
	}

	return 0;
}