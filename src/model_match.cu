#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "math_constants.h"
#include "device_functions.h"
#include<device_launch_parameters.h>
#include <thrust/version.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/extrema.h"
#include <chrono>  // for high_resolution_clock

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void TransformPointsKernel(
    float* pPointModels,		   // size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointPerModel,
	int batchSize,
    float* pPoses,				// size: batchSize * 16 ;; t1[16], t2[16], ... 						
	float* pPointModelTranses    // size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	)
{

	//************input check *******************//
	if (
		pPointModels == NULL
		|| pPointModelTranses == NULL
		|| pPoses == NULL
		)
	{
		return;
	}

	int gii = blockDim.x * blockIdx.x + threadIdx.x; // the ?th points, one point one thread
	int totalSize = numOfPointPerModel * batchSize;
	if (gii >= totalSize)
	{
		return;
	}

	int pointIdx = gii;
	float x = pPointModels[(pointIdx * 3)];
	float y = pPointModels[(pointIdx * 3) + 1];
	float z = pPointModels[(pointIdx * 3) + 2];

	int poseIdx = pointIdx / numOfPointPerModel;

	pPointModelTranses[(pointIdx * 3)] = pPoses[poseIdx * 16 + 0] * x + pPoses[poseIdx * 16 + 1] * y + pPoses[poseIdx * 16 + 2] * z + pPoses[poseIdx * 16 + 3];
	pPointModelTranses[(pointIdx * 3) + 1] = pPoses[poseIdx * 16 + 4] * x + pPoses[poseIdx * 16 + 5] * y + pPoses[poseIdx * 16 + 6] * z + pPoses[poseIdx * 16 + 7];
	pPointModelTranses[(pointIdx * 3) + 2] = pPoses[poseIdx * 16 + 8] * x + pPoses[poseIdx * 16 + 9] * y + pPoses[poseIdx * 16 + 10] * z + pPoses[poseIdx * 16 + 11];

	//pPointModelTranses[(pointIdx * 3)] = 1;
	//pPointModelTranses[(pointIdx * 3) + 1] = 1;
	//pPointModelTranses[(pointIdx * 3) + 2] = 1;

}

int  TransformPointsCU(
	float* pPointModels,		// size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointPerModel,
	int batchSize,
    float* pPoses,				// size: batchSize * 16 ;; t1[16], t2[16], ... 					
	float* pPointModelTranses     // size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	)
{
	unsigned int threadsPerBlock = 1024;
	unsigned int totalThreadSize = numOfPointPerModel * batchSize;
	unsigned int blocksPerGrid = (totalThreadSize + threadsPerBlock - 1) / threadsPerBlock;
	TransformPointsKernel << <blocksPerGrid, threadsPerBlock >> > (
		pPointModels,
		numOfPointPerModel,
		batchSize,
		pPoses,
		pPointModelTranses);

    gpuErrchk( cudaPeekAtLastError());
	gpuErrchk(cudaThreadSynchronize());

	return 0;
}

__global__ void findAndVerifyNearestPointsKernel(
    float* pPointModels,		// size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointPerModel,
	float *pPointScene,			// size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointScene,
	int batchSize,
	float distThd,
    int* ppointsValids     // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	)
{
	//************input check *******************//
	if (
		pPointModels == NULL
		|| pPointScene == NULL
		|| ppointsValids == NULL
		)
	{
		return;
	}

	int gii = blockDim.x * blockIdx.x + threadIdx.x; // the ?th points, one point one thread
	int totalSize = numOfPointPerModel * batchSize;
	if (gii >= totalSize)
	{
		return;
	}

	int pointIdx = gii;
	float x = pPointModels[(pointIdx * 3)];
	float y = pPointModels[(pointIdx * 3) + 1];
	float z = pPointModels[(pointIdx * 3) + 2];

	//float minSqDist = 100; // mm2
	//float minSIdx = 0;
	for (int sceneIdx = 0; sceneIdx < numOfPointScene; ++sceneIdx)
	{
		float x_scene = pPointScene[sceneIdx * 3];
		float y_scene = pPointScene[sceneIdx * 3 + 1];
		float z_scene = pPointScene[sceneIdx * 3 + 2];
		float sqdist = (x - x_scene) * (x - x_scene) + (y - y_scene) *  (y - y_scene) + (z - z_scene) *  (z - z_scene);
		
		if (sqdist < distThd)
		{
			ppointsValids[pointIdx] = 1;
			return;
		}
	}
	
	ppointsValids[pointIdx] = 0;
	return;
}

int  findAndVerifyNearestPointsCU(
	float* pPointModels,		// size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointPerModel,
	float *pPointScene,			// size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointScene,
	int batchSize,
	float distThd,
    int* ppointsValids     // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	)
{
	
	unsigned int threadsPerBlock = 1024;
	unsigned int totalThreadSize = numOfPointPerModel * batchSize;
	unsigned int blocksPerGrid = (totalThreadSize + threadsPerBlock - 1) / threadsPerBlock;
	findAndVerifyNearestPointsKernel << <blocksPerGrid, threadsPerBlock >> > (
		pPointModels,
		numOfPointPerModel,
		pPointScene,
		numOfPointScene,
		batchSize,
		distThd,
		ppointsValids);

    gpuErrchk( cudaPeekAtLastError());
	gpuErrchk(cudaThreadSynchronize());


	return 0;
}

__global__ void verifyPointsNearKernel(
    float* pdists,        // size: numOfPointPerModel * batchSize ;;  t1 pt1 dist, t1 pt2 dist, ...  , t1 ptn dist, t2 pt1 dist, t2 pt2 dist, ...... 
	int numOfPointPerModel,
	int batchSize,
	float distThd,
    int* ppointsValids     // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	)
{
	//************input check *******************//
	if (
		pdists == NULL
		|| ppointsValids == NULL
		)
	{
		return;
	}

	int gii = blockDim.x * blockIdx.x + threadIdx.x; // the ?th points, one point one thread
	int totalSize = numOfPointPerModel * batchSize;
	if (gii >= totalSize)
	{
		return;
	}

	int pointIdx = gii;

	float dist = pdists[pointIdx];
	ppointsValids[pointIdx] = 0;
	if (dist < distThd)
	{
		ppointsValids[pointIdx] = 1;
	}
}

int  verifyPointsNearCU(
	float* pdists,        // size: numOfPointPerModel * batchSize ;;  t1 pt1 dist, t1 pt2 dist, ...  , t1 ptn dist, t2 pt1 dist, t2 pt2 dist, ...... 
	int numOfPointPerModel,
	int batchSize,
	float distThd,
    int* ppointsValids     // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	)
{
	
	unsigned int threadsPerBlock = 1024;
	unsigned int totalThreadSize = numOfPointPerModel * batchSize;
	unsigned int blocksPerGrid = (totalThreadSize + threadsPerBlock - 1) / threadsPerBlock;
	verifyPointsNearKernel << <blocksPerGrid, threadsPerBlock >> > (
		pdists,
		numOfPointPerModel,
		batchSize,
		distThd,
		ppointsValids);

    gpuErrchk( cudaPeekAtLastError());
	gpuErrchk(cudaThreadSynchronize());


	return 0;
}


__global__ void computePoseLCPKernel(
    int* ppointsValids, // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	int numOfPointPerModel,
	int batchSize,
	float * pLCPs   // size: batchSize;; LCP1, LCP2, ... LCPn	
	)
{
	//************input check *******************//
	if (
		pLCPs == NULL
		|| ppointsValids == NULL
		)
	{
		return;
	}

	int gii = blockDim.x * blockIdx.x + threadIdx.x; // the ?th pose, one pose one thread
	int totalSize = batchSize;
	if (gii >= totalSize)
	{
		return;
	}

	int IdxInBatch = gii;

	int count = 0;
	for (int i = 0 ; i < numOfPointPerModel; ++i)
	{
		count = count + ppointsValids[IdxInBatch * numOfPointPerModel + i];
	}
	pLCPs[IdxInBatch] = (float)count/(float)numOfPointPerModel;
}

int  computePoseLCP_CU(
	int* ppointsValids,   // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	int numOfPointPerModel,
	int batchSize,
	float * pLCPs      // size: batchSize;; LCP1, LCP2, ... LCPn	
	)
{
	unsigned int threadsPerBlock = 1024;
	unsigned int totalThreadSize = batchSize;
	unsigned int blocksPerGrid = (totalThreadSize + threadsPerBlock - 1) / threadsPerBlock;
	computePoseLCPKernel << <blocksPerGrid, threadsPerBlock >> > (
		ppointsValids,
		numOfPointPerModel,
		batchSize,
		pLCPs);

    gpuErrchk( cudaPeekAtLastError());
	gpuErrchk(cudaThreadSynchronize());

	return 0;
}

