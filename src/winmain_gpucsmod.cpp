#include <filesystem>
#include <algorithm>
#include "rgbd.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "model_match.cuh"
#include <memory>

namespace fs = std::experimental::filesystem;

// alg parameters
extern const float ds_voxel_size;
extern const float normal_radius;
extern const int ppf_tr_discretization;
extern const int ppf_rot_discretization;
extern const float validPairMinDist;

extern const float lcp_distthreshold;
extern const int goodPairsMax;

// input parameters
extern const std::string scene_scale;

// running parameters
extern const int batchSize;
extern const int debug_flag;

class gpucsmod
{
public:
	gpucsmod();
	~gpucsmod();
	int init(std::string object_path, std::string ppf_path);
	int run(std::string scene_path);
private:
	bool isInited;

	// parameters
	int partitionNumberX = 4;
	int partitionNumberY = 4;
	int partitionNumberZ = 4;

	// general resources
	std::vector<Point3D> m_point3d_model;
	PPFMapType m_model_map;

	// resources for transform computing
	float* m_pPointModel;
	float* m_pPointModelNormal;
	float* m_d_pPointModelGPU;
	float* m_d_pPointModelNormalGPU;
	float* m_d_pPosesGPU_Batch;
	int* m_pSrcId0_Batch;
	int* m_pSrcId1_Batch;
	int* m_pTarId0_Batch;
	int* m_pTarId1_Batch;
	int* m_d_pSrcId0_GPUBatch;
	int* m_d_pSrcId1_GPUBatch;
	int* m_d_pTarId0_GPUBatch;
	int* m_d_pTarId1_GPUBatch;

	//resources for transform verifiy
	float* m_pPointModelBatch;
	float* m_d_pPointModelGPUBatch;
	float* m_d_pPointModelTransGPUBatch;
	float* m_pPosesBatch;
	int* m_d_ppointsValidsGPUBatch;
	float* m_d_pLCPsGPUBatch;

};


gpucsmod::gpucsmod()
{
	isInited = false;
}

gpucsmod::~gpucsmod()
{
	isInited = false;

	//free resources for transform computing, need to add null check
	delete[] m_pPointModel;
	delete[] m_pPointModelNormal;
	cudaFree(m_d_pPointModelGPU);
	cudaFree(m_d_pPointModelNormalGPU);
	cudaFree(m_d_pPosesGPU_Batch);
	delete[] m_pSrcId0_Batch;
	delete[] m_pSrcId1_Batch;
	delete[] m_pTarId0_Batch;
	delete[] m_pTarId1_Batch;
	cudaFree(m_d_pSrcId0_GPUBatch);
	cudaFree(m_d_pSrcId1_GPUBatch);
	cudaFree(m_d_pTarId0_GPUBatch);
	cudaFree(m_d_pTarId1_GPUBatch);

	//free resources for transform verify, need to add null check
	delete[] m_pPosesBatch;
	delete[] m_pPointModelBatch;
	cudaFree(m_d_pPointModelGPUBatch);
	cudaFree(m_d_pPointModelTransGPUBatch);
	cudaFree(m_d_pLCPsGPUBatch);
}

int gpucsmod::init(std::string object_path, std::string ppf_path)
{
	if (isInited == true)
	{
		std::cout << "Already inited.\n";
		return -1;
	}

	std::string model_map_path = ppf_path;
	std::string model_location = object_path;
	std::string debug_path = "./dbg";
	fs::create_directories(debug_path);

	auto start = std::chrono::high_resolution_clock::now();
	/***********  load PPF map ********************/
	m_model_map.clear();
	rgbd::load_ppf_map(model_map_path, m_model_map);

	/********** load model ********************************/
	m_point3d_model.clear();
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::io::loadPLYFile(model_location, *cloud);
	rgbd::load_ply_model(cloud, m_point3d_model, 1.0f);

	if (debug_flag == 1)
	{
		rgbd::save_as_ply(debug_path + "/model.ply", m_point3d_model, 1.0);
	}
	/********** print info ********************************/
	std::cout << "|M| = " << m_point3d_model.size() << ",  |map(M)| = " << m_model_map.size() << std::endl;

	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << "Loading model" << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	start = std::chrono::high_resolution_clock::now();
	/********** malloc for  transform computing********************************/
	size_t number_of_points_model = m_point3d_model.size();
	m_pPointModel = new float[number_of_points_model * 3];
	for (int i = 0; i < number_of_points_model; i++)
	{
		m_pPointModel[i * 3 + 0] = m_point3d_model[i].x();
		m_pPointModel[i * 3 + 1] = m_point3d_model[i].y();
		m_pPointModel[i * 3 + 2] = m_point3d_model[i].z();
	}
	m_pPointModelNormal = new float[number_of_points_model * 3];
	for (int i = 0; i < number_of_points_model; i++)
	{
		m_pPointModelNormal[i * 3 + 0] = m_point3d_model[i].normal().x();
		m_pPointModelNormal[i * 3 + 1] = m_point3d_model[i].normal().y();
		m_pPointModelNormal[i * 3 + 2] = m_point3d_model[i].normal().z();
	}

	cudaMalloc((void**)&m_d_pPointModelGPU, sizeof(float)* number_of_points_model * 3);
	cudaMemcpy(m_d_pPointModelGPU, m_pPointModel, sizeof(float)* number_of_points_model * 3, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_d_pPointModelNormalGPU, sizeof(float)* number_of_points_model * 3);
	cudaMemcpy(m_d_pPointModelNormalGPU, m_pPointModelNormal, sizeof(float)* number_of_points_model * 3, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_d_pPosesGPU_Batch, sizeof(float)* batchSize * 16);

	m_pSrcId0_Batch = new int[batchSize];
	m_pSrcId1_Batch = new int[batchSize];
	m_pTarId0_Batch = new int[batchSize];
	m_pTarId1_Batch = new int[batchSize];

	cudaMalloc((void**)&m_d_pSrcId0_GPUBatch, sizeof(int)* batchSize);
	cudaMalloc((void**)&m_d_pSrcId1_GPUBatch, sizeof(int)* batchSize);
	cudaMalloc((void**)&m_d_pTarId0_GPUBatch, sizeof(int)* batchSize);
	cudaMalloc((void**)&m_d_pTarId1_GPUBatch, sizeof(int)* batchSize);


	/********** malloc for  transform verify********************************/
	m_pPointModelBatch = new float[number_of_points_model * 3 * batchSize];
	cudaMalloc((void**)&m_d_pPointModelGPUBatch, sizeof(float)* number_of_points_model * 3 * batchSize);
	cudaMalloc((void**)&m_d_pPointModelTransGPUBatch, sizeof(float)* number_of_points_model * 3 * batchSize);
	m_pPosesBatch = new float[16 * batchSize];

	cudaMalloc((void**)&m_d_ppointsValidsGPUBatch, sizeof(int)* number_of_points_model * batchSize);
	cudaMalloc((void**)&m_d_pLCPsGPUBatch, sizeof(float)* batchSize);

	/*** assign points***/
	for (int j = 0; j < batchSize; ++j)
	{
		for (int i = 0; i < number_of_points_model; i++)
		{
			m_pPointModelBatch[j * number_of_points_model * 3 + i * 3 + 0] = m_point3d_model[i].x();
			m_pPointModelBatch[j * number_of_points_model * 3 + i * 3 + 1] = m_point3d_model[i].y();
			m_pPointModelBatch[j * number_of_points_model * 3 + i * 3 + 2] = m_point3d_model[i].z();
		}
	}
	cudaMemcpy(m_d_pPointModelGPUBatch, m_pPointModelBatch, sizeof(float)* number_of_points_model * batchSize * 3, cudaMemcpyHostToDevice);

	finish = std::chrono::high_resolution_clock::now();
	std::cout << "malloc model" << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";
	return 0;
}

int gpucsmod::run(std::string scene_path)
{
	std::string x_location = scene_path + "/XMap.tif";
	std::string y_location = scene_path + "/YMap.tif";
	std::string z_location = scene_path + "/ZMap.tif";
	std::string isValid_location = scene_path + "/isValid.bmp";
	std::string segmentation_map_location = scene_path + "/segmentation.bmp";
	std::string debug_path = "./dbg";
	fs::create_directories(debug_path);

	/***********  load scene in sample form ********************/
	auto start = std::chrono::high_resolution_clock::now();
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

	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << "Loading scene " << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	std::cout << "|S| = " << point3d_scene.size() << std::endl;
	if (debug_flag == 1)
	{
		rgbd::save_as_ply(debug_path + "/sampled_scene.ply", point3d_scene, 1.0);
	}

	/***********  calculate ppf pairs********************/
	start = std::chrono::high_resolution_clock::now();
	std::vector<std::pair<int, int>> goodPairs;
	for (int i = 0; i < point3d_scene.size(); i++)
	{
		for (int j = i + 1; j < point3d_scene.size(); j++)
		{
			Point3D p1 = point3d_scene[i];
			Point3D p2 = point3d_scene[j];

			float dist = std::sqrt(
				(p1.x() - p2.x()) * (p1.x() - p2.x())
				+ (p1.y() - p2.y()) * (p1.y() - p2.y())
				+ (p1.z() - p2.z()) * (p1.z() - p2.z()));

			if (dist < validPairMinDist)
			{
				continue;
			}
			goodPairs.push_back(std::make_pair(i, j));
		}
	}

	finish = std::chrono::high_resolution_clock::now();
	std::cout << "obtain ppf pairs " << goodPairs.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	/***********  shuffle ********************/

	start = std::chrono::high_resolution_clock::now();
	std::random_shuffle(goodPairs.begin(), goodPairs.end());
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "random_shuffle " << goodPairs.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	std::cout << "use at most " << std::min(goodPairs.size(), (size_t)goodPairsMax) << " pairs.\n";

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

		auto map_it = m_model_map.find(ppf);
		if (map_it != m_model_map.end())
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
	std::cout << "calculate correspondences " << poseEsts.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";

	// save readable format
	if (debug_flag == 1)
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

		/***  allocate memory ***/
		float* pAllPoses = new float[16 * totalComputeSize];
		memset(pAllPoses, 0, sizeof(float) * 16 * totalComputeSize);

		size_t number_of_points_model = m_point3d_model.size();

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
		
		float* d_pPointSceneGPU;
		cudaMalloc((void**)&d_pPointSceneGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointSceneGPU, pPointScene, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);

		float* d_pPointSceneNormalGPU;
		cudaMalloc((void**)&d_pPointSceneNormalGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointSceneNormalGPU, pPointSceneNormal, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);

		/*** run loop ***/
		start = std::chrono::high_resolution_clock::now();

		for (int k = 0; k < numOfBatch; ++k)
		{
			for (int i = 0; i < batchSize; ++i)
			{
				int currentPoseIdx = k * batchSize + i;
				if (currentPoseIdx < totalPoseSize)
				{
					m_pSrcId0_Batch[i] = poseEsts.at(k * batchSize + i).srcId[0];
					m_pSrcId1_Batch[i] = poseEsts.at(k * batchSize + i).srcId[1];
					m_pTarId0_Batch[i] = poseEsts.at(k * batchSize + i).tarId[0];
					m_pTarId1_Batch[i] = poseEsts.at(k * batchSize + i).tarId[1];
				}
			}
			cudaMemcpy(m_d_pSrcId0_GPUBatch, m_pSrcId0_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(m_d_pSrcId1_GPUBatch, m_pSrcId1_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(m_d_pTarId0_GPUBatch, m_pTarId0_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(m_d_pTarId1_GPUBatch, m_pTarId1_Batch, sizeof(int)* batchSize, cudaMemcpyHostToDevice);

			ComputeTransformForCorrespondencesCU(
				m_d_pPointModelGPU,
				m_d_pPointModelNormalGPU,
				number_of_points_model,
				d_pPointSceneGPU,
				d_pPointSceneNormalGPU,
				number_of_points_scene,
				m_d_pSrcId0_GPUBatch,
				m_d_pSrcId1_GPUBatch,
				m_d_pTarId0_GPUBatch,
				m_d_pTarId1_GPUBatch,
				batchSize,
				m_d_pPosesGPU_Batch
			);
			cudaMemcpy(pAllPoses + (k * batchSize * 16), m_d_pPosesGPU_Batch, sizeof(float) * batchSize * 16, cudaMemcpyDeviceToHost);
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

		
		delete[] pPointScene;
		delete[] pPointSceneNormal;
		cudaFree(d_pPointSceneGPU);
		cudaFree(d_pPointSceneNormalGPU);
		delete[] pAllPoses;
	}

	if (debug_flag == 1)
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
		/*** built ************/
		/***  reorder scene points ***/
		std::vector<Point3D> point3d_scene_sorted;
		std::vector<int> vnumOfpointsPerPartition;
		std::vector<int> vPartitionStartIdx;
		vnumOfpointsPerPartition.resize(partitionNumberX * partitionNumberY * partitionNumberZ);
		vPartitionStartIdx.resize(partitionNumberX * partitionNumberY * partitionNumberZ);

		/***  find max min scene points ***/
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

		/***  extend bbox ***/
		maxPt.x() += 2 * lcp_distthreshold;
		maxPt.y() += 2 * lcp_distthreshold;
		maxPt.z() += 2 * lcp_distthreshold;
		minPt.x() -= 2 * lcp_distthreshold;
		minPt.y() -= 2 * lcp_distthreshold;
		minPt.z() -= 2 * lcp_distthreshold;

		/***  get partitionSize ***/
		float partitionSizeX = (maxPt.x() - minPt.x()) / partitionNumberX;
		float partitionSizeY = (maxPt.y() - minPt.y()) / partitionNumberY;
		float partitionSizeZ = (maxPt.z() - minPt.z()) / partitionNumberZ;

		/***  classify scene points into voxel ***/
		std::vector<std::vector<int>> AllScenePointsIndexInVoxel;
		AllScenePointsIndexInVoxel.resize(partitionNumberX * partitionNumberY * partitionNumberZ);
		for (int i = 0; i < point3d_scene.size(); i++)
		{
			float x = point3d_scene[i].x();
			float y = point3d_scene[i].y();
			float z = point3d_scene[i].z();

			int xId = (x - minPt.x()) / partitionSizeX;
			int yId = (y - minPt.y()) / partitionSizeY;
			int zId = (z - minPt.z()) / partitionSizeZ;

			int hashId = zId * partitionNumberY * partitionNumberX + yId * partitionNumberX + xId;
			AllScenePointsIndexInVoxel[hashId].push_back(i);
		}

		/***  obtain  sorted scene points and relevant structure ***/
		int curIdx = 0;
		for (int i = 0; i < AllScenePointsIndexInVoxel.size(); ++i)
		{
			vPartitionStartIdx[i] = curIdx;
			for (int j = 0; j < AllScenePointsIndexInVoxel[i].size(); ++j)
			{
				point3d_scene_sorted.push_back(point3d_scene[AllScenePointsIndexInVoxel[i][j]]);
				curIdx++;
			}
			vnumOfpointsPerPartition[i] = AllScenePointsIndexInVoxel[i].size();
		}
		std::cout << " point3d_scene_sorted size = " << point3d_scene_sorted.size() << "\n";

		// logging
		if (debug_flag == 1)
		{
			std::string path;
			path = debug_path + "/voxelPartition.txt";
			std::FILE* f = std::fopen(path.c_str(), "w");

			int count = 0;
			for (int i = 0; i < AllScenePointsIndexInVoxel.size(); ++i)
			{
				count += AllScenePointsIndexInVoxel[i].size();
			}
			std::fprintf(f, "===================================================\n");
			std::fprintf(f, "total %d points\n", count);

			std::fprintf(f, "minX = %.3f maxX = %.3f\n", minPt.x(), maxPt.x());
			std::fprintf(f, "minY = %.3f maxY = %.3f\n", minPt.y(), maxPt.y());
			std::fprintf(f, "minZ = %.3f maxZ = %.3f\n", minPt.z(), maxPt.z());

			std::fprintf(f, "parSizeX = %.3f parNumX = %d\n", partitionSizeX, partitionNumberX);
			std::fprintf(f, "parSizeY = %.3f parNumY = %d\n", partitionSizeY, partitionNumberY);
			std::fprintf(f, "parSizeZ = %.3f parNumZ = %d\n", partitionSizeZ, partitionNumberZ);

			std::vector<std::pair<Point3D, Point3D>> voxelminmax;
			for (int i = 0; i < partitionNumberZ; ++i)
			{
				float zmin = minPt.z() + i * partitionSizeZ;
				float zmax = minPt.z() + (i + 1) * partitionSizeZ;
				for (int j = 0; j < partitionNumberY; ++j)
				{
					float ymin = minPt.y() + j * partitionSizeY;
					float ymax = minPt.y() + (j + 1) * partitionSizeY;
					for (int k = 0; k < partitionNumberX; ++k)
					{
						float xmin = minPt.x() + k * partitionSizeX;
						float xmax = minPt.x() + (k + 1) * partitionSizeX;
						std::pair<Point3D, Point3D> ptpair;
						Point3D p1;
						p1.x() = xmin;
						p1.y() = ymin;
						p1.z() = zmin;
						Point3D p2;
						p2.x() = xmax;
						p2.y() = ymax;
						p2.z() = zmax;
						ptpair.first = p1;
						ptpair.second = p2;
						voxelminmax.push_back(ptpair);
					}
				}
			}

			for (int i = 0; i < voxelminmax.size(); ++i)
			{
				std::fprintf(f, "{%.3f,%.3f,%.3f} to {%.3f,%.3f,%.3f} :: %d points :: ",
					voxelminmax[i].first.x(), voxelminmax[i].first.y(), voxelminmax[i].first.z(),
					voxelminmax[i].second.x(), voxelminmax[i].second.y(), voxelminmax[i].second.z(), AllScenePointsIndexInVoxel[i].size());

				for (int j = 0; j < AllScenePointsIndexInVoxel[i].size(); ++j)
				{
					std::fprintf(f, "{%.3f,%.3f,%.3f}",
						point3d_scene[AllScenePointsIndexInVoxel[i][j]].x(),
						point3d_scene[AllScenePointsIndexInVoxel[i][j]].y(),
						point3d_scene[AllScenePointsIndexInVoxel[i][j]].z());
				}
				std::fprintf(f, "\n");
			}

			std::fprintf(f, "===================================================\n");
			for (int i = 0; i < vPartitionStartIdx.size(); ++i)
			{
				std::fprintf(f, "PartitionStartIdx = %d\n", vPartitionStartIdx[i]);
				for (int j = vPartitionStartIdx[i]; j < (vPartitionStartIdx[i] + vnumOfpointsPerPartition[i]); ++j)
				{
					std::fprintf(f, "{%d : {%.3f,%.3f,%.3f}}", j,
						point3d_scene_sorted[j].x(), point3d_scene_sorted[j].y(), point3d_scene_sorted[j].z());
				}
				std::fprintf(f, "\n", vPartitionStartIdx[i]);
			}
			std::fclose(f);
		}

		// logging
		if (debug_flag == 1)
		{
			for (int i = 0; i < vPartitionStartIdx.size(); ++i)
			{
				std::vector<Point3D> point3d_scene_segment;
				for (int j = vPartitionStartIdx[i]; j < (vPartitionStartIdx[i] + vnumOfpointsPerPartition[i]); ++j)
				{
					point3d_scene_segment.push_back(point3d_scene_sorted[j]);
				}
				char buffer[1024];
				snprintf(buffer, 1024, "/scene_seg_%d.ply", i);
				rgbd::save_as_ply(debug_path + buffer, point3d_scene_segment, 1);
			}
		}

		/***  malloc resources ***/
		size_t number_of_points_scene = point3d_scene_sorted.size();
		float* pPointScene = new float[number_of_points_scene * 3];
		for (int i = 0; i < number_of_points_scene; i++)
		{
			pPointScene[i * 3 + 0] = point3d_scene_sorted[i].x();
			pPointScene[i * 3 + 1] = point3d_scene_sorted[i].y();
			pPointScene[i * 3 + 2] = point3d_scene_sorted[i].z();
		}

		float* d_pPointSceneGPU;
		cudaMalloc((void**)&d_pPointSceneGPU, sizeof(float)* number_of_points_scene * 3);
		cudaMemcpy(d_pPointSceneGPU, pPointScene, sizeof(float)* number_of_points_scene * 3, cudaMemcpyHostToDevice);

		int* pNumOfPointSceneInPartition = &vnumOfpointsPerPartition[0];
		int* d_pNumOfPointSceneInPartition;
		cudaMalloc((void**)&d_pNumOfPointSceneInPartition, sizeof(int)* vnumOfpointsPerPartition.size());
		cudaMemcpy(d_pNumOfPointSceneInPartition, pNumOfPointSceneInPartition, sizeof(int)* vnumOfpointsPerPartition.size(), cudaMemcpyHostToDevice);

		int* pPartitionStartIdx = &vPartitionStartIdx[0];
		int* d_pPartitionStartIdx;
		cudaMalloc((void**)&d_pPartitionStartIdx, sizeof(int)* vPartitionStartIdx.size());
		cudaMemcpy(d_pPartitionStartIdx, pPartitionStartIdx, sizeof(int)* vPartitionStartIdx.size(), cudaMemcpyHostToDevice);

		float p_minPt[3] = { minPt.x(), minPt.y(), minPt.z() };
		float p_maxPt[3] = { maxPt.x(), maxPt.y(), maxPt.z() };
		float p_partitionSize[3] = { partitionSizeX, partitionSizeY, partitionSizeZ };
		int p_partitionNumber[3] = { partitionNumberX, partitionNumberY, partitionNumberZ };

		float* d_p_minPt;
		float* d_p_maxPt;
		float* d_p_partitionSize;
		int* d_p_partitionNumber;
		cudaMalloc((void**)&d_p_minPt, sizeof(float) * 3);
		cudaMemcpy(d_p_minPt, p_minPt, sizeof(float) * 3, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_p_maxPt, sizeof(float) * 3);
		cudaMemcpy(d_p_maxPt, p_maxPt, sizeof(float) * 3, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_p_partitionSize, sizeof(float) * 3);
		cudaMemcpy(d_p_partitionSize, p_partitionSize, sizeof(float) * 3, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_p_partitionNumber, sizeof(int) * 3);
		cudaMemcpy(d_p_partitionNumber, p_partitionNumber, sizeof(int) * 3, cudaMemcpyHostToDevice);

		int numberOfBatch = poseEsts.size() / batchSize + 1;
		int totalPoseNum = poseEsts.size();
		int totalComputeNum = numberOfBatch * batchSize;
		float* pLCPs = new float[totalComputeNum];
		memset(pLCPs, 0, sizeof(float) * totalComputeNum);

		size_t number_of_points_model = m_point3d_model.size();
		/***  run ***/
		{
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
							m_pPosesBatch[j * 16 + i] = poseEsts[k * batchSize + j].trans44(row, col);
						}
					}
				}
				cudaMemcpy(m_d_pPosesGPU_Batch, m_pPosesBatch, sizeof(float) * 16 * batchSize, cudaMemcpyHostToDevice);
				TransformPointsCU(
					m_d_pPointModelGPUBatch,
					number_of_points_model,
					batchSize,
					m_d_pPosesGPU_Batch,
					m_d_pPointModelTransGPUBatch
				);

				float sqdistThd = lcp_distthreshold * lcp_distthreshold; // since flann dist is sqdist
				findAndVerifyNearestPointsVoxelPartitionCU(
					m_d_pPointModelTransGPUBatch,
					number_of_points_model,
					d_pPointSceneGPU,
					number_of_points_scene,
					d_pNumOfPointSceneInPartition,
					d_pPartitionStartIdx,
					d_p_partitionSize,
					d_p_partitionNumber,
					d_p_minPt,
					d_p_maxPt,
					batchSize,
					sqdistThd,
					m_d_ppointsValidsGPUBatch);

				computePoseLCP_CU(
					m_d_ppointsValidsGPUBatch,
					number_of_points_model,
					batchSize,
					m_d_pLCPsGPUBatch);

				cudaMemcpy(pLCPs + (k * batchSize), m_d_pLCPsGPUBatch, sizeof(float) * batchSize, cudaMemcpyDeviceToHost);
			}

			float* maxAddr = std::max_element(pLCPs, pLCPs + totalPoseNum);
			int maxLCPIdx = std::distance(pLCPs, maxAddr);
			std::cout << "max LCP at: " << maxLCPIdx << " , LCP = " << pLCPs[maxLCPIdx] << '\n';

			best_index = maxLCPIdx;

			finish = std::chrono::high_resolution_clock::now();
			std::cout << "verify transform " << poseEsts.size() << " in "
				<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
				<< " milliseconds\n";
		}
		if (debug_flag == 1)
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

			rgbd::transform_pointset(m_point3d_model, point3d_model_pose, poseEsts[best_index].trans44);
			rgbd::save_as_ply(debug_path + "/best_pose.ply", point3d_model_pose, 1);
			rgbd::save_as_ply(debug_path + "/scene.ply", point3d_scene, 1);

			if (scene_scale == "mm")
			{
				rgbd::save_as_ply(debug_path + "/best_pose_1000.ply", point3d_model_pose, 1000);
			}
		}

		// logging
		if (debug_flag == 1)
		{
			// assign pose to GPU
			int currentPoseIdx = best_index;
			for (int i = 0; i < 16; i++)
			{
				int row = i / 4;
				int col = i % 4;
				m_pPosesBatch[i] = poseEsts[currentPoseIdx].trans44(row, col);
			}

			cudaMemcpy(m_d_pPosesGPU_Batch, m_pPosesBatch, sizeof(float) * 16 * batchSize, cudaMemcpyHostToDevice);

			TransformPointsCU(
				m_d_pPointModelGPUBatch,
				number_of_points_model,
				batchSize,
				m_d_pPosesGPU_Batch,
				m_d_pPointModelTransGPUBatch
			);

			float sqdistThd = lcp_distthreshold * lcp_distthreshold; // since flann dist is sqdist
			findAndVerifyNearestPointsVoxelPartitionCU(
				m_d_pPointModelTransGPUBatch,
				number_of_points_model,
				d_pPointSceneGPU,
				number_of_points_scene,
				d_pNumOfPointSceneInPartition,
				d_pPartitionStartIdx,
				d_p_partitionSize,
				d_p_partitionNumber,
				d_p_minPt,
				d_p_maxPt,
				batchSize,
				sqdistThd,
				m_d_ppointsValidsGPUBatch);

			int* ppointsValidsGPU_DEBUG = new int[number_of_points_model * batchSize];
			cudaMemcpy(ppointsValidsGPU_DEBUG, m_d_ppointsValidsGPUBatch, sizeof(int)* number_of_points_model * batchSize, cudaMemcpyDeviceToHost);

			std::string path;
			path = debug_path + "/isValidBest.txt";
			std::FILE* f = std::fopen(path.c_str(), "w");

			for (int i = 0; i < number_of_points_model; ++i)
			{
				std::fprintf(f, "%d\n", ppointsValidsGPU_DEBUG[i]);
			}
			std::fclose(f);

			std::vector<Point3D> point3d_model_best_inliner;
			std::vector<Point3D> point3d_model_best_outliner;
			for (int i = 0; i < m_point3d_model.size(); ++i)
			{
				if (ppointsValidsGPU_DEBUG[i] > 0)
				{
					point3d_model_best_inliner.push_back(m_point3d_model[i]);
				}
				else
				{
					point3d_model_best_outliner.push_back(m_point3d_model[i]);
				}
			}
			std::vector<Point3D> point3d_model_best_inliner_trans;
			std::vector<Point3D> point3d_model_best_outliner_trans;
			rgbd::transform_pointset(point3d_model_best_inliner, point3d_model_best_inliner_trans, poseEsts[best_index].trans44);
			rgbd::transform_pointset(point3d_model_best_outliner, point3d_model_best_outliner_trans, poseEsts[best_index].trans44);
			rgbd::save_as_ply(debug_path + "/best_pose_inliner.ply", point3d_model_best_inliner_trans, 1);
			rgbd::save_as_ply(debug_path + "/best_pose_outliner.ply", point3d_model_best_outliner_trans, 1);
		}


		/********** free ************/
		delete[] pPointScene;
		delete[] pLCPs;
		cudaFree(d_pPartitionStartIdx);
		cudaFree(d_pNumOfPointSceneInPartition);
		cudaFree(d_p_minPt);
		cudaFree(d_p_maxPt);
		cudaFree(d_p_partitionSize);
		cudaFree(d_p_partitionNumber);
	}

	return 0;
}


int gpucs7(std::string scene_path, std::string object_path, std::string ppf_path)
{
	std::shared_ptr<gpucsmod> p = std::make_shared< gpucsmod>();
	p->init(object_path, ppf_path);

	for (int speedTestIdx = 0; speedTestIdx < 10; ++speedTestIdx)
	{
		auto start = std::chrono::high_resolution_clock::now();
		p->run(scene_path);
		auto finish = std::chrono::high_resolution_clock::now();
		std::cout << "gpucs" << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";
	}
	return 0;
}