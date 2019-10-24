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



__device__ void mul33(float mat1[9],float mat2[9], float matout[9])
{
	matout[0] = mat1[0] * mat2[0] + mat1[1] * mat2[3] + mat1[2] * mat2[6];
	matout[1] = mat1[0] * mat2[1] + mat1[1] * mat2[4] + mat1[2] * mat2[7];
	matout[2] = mat1[0] * mat2[2] + mat1[1] * mat2[5] + mat1[2] * mat2[8];

	matout[3] = mat1[3] * mat2[0] + mat1[4] * mat2[3] + mat1[5] * mat2[6];
	matout[4] = mat1[3] * mat2[1] + mat1[4] * mat2[4] + mat1[5] * mat2[7];
	matout[5] = mat1[3] * mat2[2] + mat1[4] * mat2[5] + mat1[5] * mat2[8];

	matout[6] = mat1[6] * mat2[0] + mat1[7] * mat2[3] + mat1[8] * mat2[6];
	matout[7] = mat1[6] * mat2[1] + mat1[7] * mat2[4] + mat1[8] * mat2[7];
	matout[8] = mat1[6] * mat2[2] + mat1[7] * mat2[5] + mat1[8] * mat2[8];
};


__device__ float3 mul3v(float mat1[9],float3 v1)
{
	return make_float3(
	mat1[0] * v1.x + mat1[1] * v1.y + mat1[2] * v1.z,
	mat1[3] * v1.x + mat1[4] * v1.y + mat1[5] * v1.z,
	mat1[6] * v1.x + mat1[7] * v1.y + mat1[8] * v1.z);
};

__device__  void transpose33(float mat1[9], float matout[9])
{
	matout[0] = mat1[0]; 
	matout[1] = mat1[3];
	matout[2] = mat1[6];

	matout[3] = mat1[1];
	matout[4] = mat1[4];
	matout[5] = mat1[7];

	matout[6] = mat1[2];
	matout[7] = mat1[5];
	matout[8] = mat1[8];
};


__device__ void mul44(float mat1[16],float mat2[16], float matout[16])
{
	matout[0] = mat1[0] * mat2[0] + mat1[1] * mat2[4] + mat1[2] * mat2[8] + mat1[3] * mat2[12] ;
	matout[1] = mat1[0] * mat2[1] + mat1[1] * mat2[5] + mat1[2] * mat2[9] + mat1[3] * mat2[13] ;
	matout[2] = mat1[0] * mat2[2] + mat1[1] * mat2[6] + mat1[2] * mat2[10] + mat1[3] * mat2[14] ;
	matout[3] = mat1[0] * mat2[3] + mat1[1] * mat2[7] + mat1[2] * mat2[11] + mat1[3] * mat2[15] ;

	matout[4] = mat1[4] * mat2[0] + mat1[5] * mat2[4] + mat1[6] * mat2[8] + mat1[7] * mat2[12] ;
	matout[5] = mat1[4] * mat2[1] + mat1[5] * mat2[5] + mat1[6] * mat2[9] + mat1[7] * mat2[13] ;
	matout[6] = mat1[4] * mat2[2] + mat1[5] * mat2[6] + mat1[6] * mat2[10] + mat1[7] * mat2[14] ;
	matout[7] = mat1[4] * mat2[3] + mat1[5] * mat2[7] + mat1[6] * mat2[11] + mat1[7] * mat2[15] ;

	matout[8] = mat1[8] * mat2[0] + mat1[9] * mat2[4] + mat1[10] * mat2[8] + mat1[11] * mat2[12] ;
	matout[9] = mat1[8] * mat2[1] + mat1[9] * mat2[5] + mat1[10] * mat2[9] + mat1[11] * mat2[13] ;
	matout[10] = mat1[8] * mat2[2] + mat1[9] * mat2[6] + mat1[10] * mat2[10] + mat1[11] * mat2[14] ;
	matout[11] = mat1[8] * mat2[3] + mat1[9] * mat2[7] + mat1[10] * mat2[11] + mat1[11] * mat2[15] ;

	matout[12] = mat1[12] * mat2[0] + mat1[13] * mat2[4] + mat1[14] * mat2[8] + mat1[15] * mat2[12] ;
	matout[13] = mat1[12] * mat2[1] + mat1[13] * mat2[5] + mat1[14] * mat2[9] + mat1[15] * mat2[13] ;
	matout[14] = mat1[12] * mat2[2] + mat1[13] * mat2[6] + mat1[14] * mat2[10] + mat1[15] * mat2[14] ;
	matout[15] = mat1[12] * mat2[3] + mat1[13] * mat2[7] + mat1[14] * mat2[11] + mat1[15] * mat2[15] ;
};

__global__ void ComputeTransformForCorrespondencesKernel(
		float* pPointModels,		// size: numOfPointModel * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		float* pPointModelsNormal,   // size: numOfPointModel * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		int numOfPointModel,
		float* pPointScene,			// size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		float* pPointSceneNormal,   // size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		int numOfPoinScene,
		int *pSrcId0,				// size: batchSize
		int *pSrcId1,				// size: batchSize
		int *pTarId0,				// size: batchSize
		int *pTarId1,				// size: batchSize
		int batchSize,						
		float* pPosesGPU     // size: batchSize * 16
	)
{
	/************
	
	input check
	
	*******************/
	if (
		pPointModels == NULL
		|| pPointModelsNormal == NULL
		|| pPointScene == NULL
		|| pPointSceneNormal == NULL
		|| pSrcId0 == NULL
		|| pSrcId1 == NULL
		|| pTarId0 == NULL
		|| pTarId1 == NULL
		|| pPosesGPU == NULL
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

	/************ 
	
	assign points 
	
	*******************/
	int srcId0 = pSrcId0[IdxInBatch];
	int srcId1 = pSrcId1[IdxInBatch];
	int tarId0 = pTarId0[IdxInBatch];
	int tarId1 = pTarId1[IdxInBatch];

	float3 src1;
	src1.x = pPointModels[srcId0 * 3 + 0];
	src1.y = pPointModels[srcId0 * 3 + 1];
	src1.z = pPointModels[srcId0 * 3 + 2];

	float3 src1normal;
	src1normal.x = pPointModelsNormal[srcId0 * 3 + 0];
	src1normal.y = pPointModelsNormal[srcId0 * 3 + 1];
	src1normal.z = pPointModelsNormal[srcId0 * 3 + 2];

	float3 src2;
	src2.x = pPointModels[srcId1 * 3 + 0];
	src2.y = pPointModels[srcId1 * 3 + 1];
	src2.z = pPointModels[srcId1 * 3 + 2];

	/*
	float3 src2normal;
	src2normal.x = pPointModelsNormal[srcId1 * 3 + 0];
	src2normal.y = pPointModelsNormal[srcId1 * 3 + 1];
	src2normal.z = pPointModelsNormal[srcId1 * 3 + 2];
	*/
	float3 tar1;
	tar1.x = pPointScene[tarId0 * 3 + 0];
	tar1.y = pPointScene[tarId0 * 3 + 1];
	tar1.z = pPointScene[tarId0 * 3 + 2];

	float3 tar1normal;
	tar1normal.x = pPointSceneNormal[tarId0 * 3 + 0];
	tar1normal.y = pPointSceneNormal[tarId0 * 3 + 1];
	tar1normal.z = pPointSceneNormal[tarId0 * 3 + 2];

	float3 tar2;
	tar2.x = pPointScene[tarId1 * 3 + 0];
	tar2.y = pPointScene[tarId1 * 3 + 1];
	tar2.z = pPointScene[tarId1 * 3 + 2];
	/*
	float3 tar2normal;
	tar2normal.x = pPointSceneNormal[tarId1 * 3 + 0];
	tar2normal.y = pPointSceneNormal[tarId1 * 3 + 1];
	tar2normal.z = pPointSceneNormal[tarId1 * 3 + 2];
	*/
	/************ 
	
	compute transform
	
	*******************/
	/************ rotsrc_normal_to_x_axis *******************/
	float theta1 = atan2(-src1normal.z, src1normal.y);
	float alpha1 = atan2(-(cos(theta1) * src1normal.y - sin(theta1) * src1normal.z), src1normal.x);
	
	float rotA1[9];
	rotA1[0] = 1;	rotA1[1] = 0;				rotA1[2] = 0; 
	rotA1[3] = 0;	rotA1[4] = cos(theta1);		rotA1[5] = -sin(theta1); 
	rotA1[6] = 0;	rotA1[7] = sin(theta1);		rotA1[8] = cos(theta1); 
	
	float rotB1[9];
	rotB1[0] = cos(alpha1);		rotB1[1] = -sin(alpha1);	 rotB1[2] = 0; 
	rotB1[3] = sin(alpha1);		rotB1[4] = cos(alpha1);		 rotB1[5] = 0; 
	rotB1[6] = 0;				rotB1[7] = 0;				 rotB1[8] = 1;
			
	float rotsrc_normal_to_x_axis[9];	
	mul33(rotB1, rotA1, rotsrc_normal_to_x_axis);

	float3 src_p2_from_p1;
	src_p2_from_p1.x = src2.x - src1.x;
	src_p2_from_p1.y = src2.y - src1.y;
	src_p2_from_p1.z = src2.z - src1.z;

	float3 src_p2_from_p1_transed = mul3v(rotsrc_normal_to_x_axis, src_p2_from_p1);
	float src_p2_from_p1_transed_angleXToXYPlane = atan2(-src_p2_from_p1_transed.z, src_p2_from_p1_transed.y);

	/************ rottar_normal_to_x_axis *******************/
	float theta2 = atan2(-tar1normal.z, tar1normal.y);
	float alpha2 = atan2(-(cos(theta2) * tar1normal.y - sin(theta2) * tar1normal.z), tar1normal.x);
	
	float rotA2[9];
	rotA2[0] = 1;	rotA2[1] = 0;				rotA2[2] = 0; 
	rotA2[3] = 0;	rotA2[4] = cos(theta2);		rotA2[5] = -sin(theta2); 
	rotA2[6] = 0;	rotA2[7] = sin(theta2);		rotA2[8] = cos(theta2); 
	
	float rotB2[9];
	rotB2[0] = cos(alpha2);		rotB2[1] = -sin(alpha2);	 rotB2[2] = 0; 
	rotB2[3] = sin(alpha2);		rotB2[4] = cos(alpha2);		 rotB2[5] = 0; 
	rotB2[6] = 0;				rotB2[7] = 0;				 rotB2[8] = 1;
			
	float rottar_normal_to_x_axis[9];	
	mul33(rotB2, rotA2, rottar_normal_to_x_axis);

	/************ rotX *******************/
	float3 tar_p2_from_p1;
	tar_p2_from_p1.x = tar2.x - tar1.x;
	tar_p2_from_p1.y = tar2.y - tar1.y;
	tar_p2_from_p1.z = tar2.z - tar1.z;

	float3 tar_p2_from_p1_transed = mul3v(rottar_normal_to_x_axis, tar_p2_from_p1);
	float tar_p2_from_p1_transed_angleXToXYPlane = atan2(-tar_p2_from_p1_transed.z, tar_p2_from_p1_transed.y);
	float angle_alpha = src_p2_from_p1_transed_angleXToXYPlane - tar_p2_from_p1_transed_angleXToXYPlane;

	float rotX[9];
	rotX[0] = 1;	rotX[1] = 0;					rotX[2] = 0; 
	rotX[3] = 0;	rotX[4] = cos(angle_alpha);		rotX[5] = -sin(angle_alpha); 
	rotX[6] = 0;	rotX[7] = sin(angle_alpha);		rotX[8] = cos(angle_alpha); 

	/************ rot *******************/
	float rot[9];
	mul33(rotX, rotsrc_normal_to_x_axis, rot);


	/************ assign transform *******************/
	float3 tslateFromSrcToOrigin;
	tslateFromSrcToOrigin.x = -src1.x;
	tslateFromSrcToOrigin.y = -src1.y;
	tslateFromSrcToOrigin.z = -src1.z;

	float rottar_normal_to_x_axis_T[9];
	transpose33(rottar_normal_to_x_axis, rottar_normal_to_x_axis_T);
	float3 tslateFromOriginToTar = tar1;

	float trans44_A[16];
	trans44_A[0] = 1;	trans44_A[1] = 0;			trans44_A[2] = 0;		trans44_A[3] = tslateFromSrcToOrigin.x; 
	trans44_A[4] = 0;	trans44_A[5] = 1;			trans44_A[6] = 0;		trans44_A[7] = tslateFromSrcToOrigin.y; 
	trans44_A[8] = 0;	trans44_A[9] = 0;			trans44_A[10] = 1;		trans44_A[11] =tslateFromSrcToOrigin.z; 
	trans44_A[12] = 0;	trans44_A[13] = 0;			trans44_A[14] = 0;		trans44_A[15] = 1; 

	float trans44_B[16];
	trans44_B[0] = rot[0];	trans44_B[1] = rot[1];			trans44_B[2] = rot[2];		trans44_B[3] =  0; 
	trans44_B[4] = rot[3];	trans44_B[5] = rot[4];			trans44_B[6] = rot[5];		trans44_B[7] =  0; 
	trans44_B[8] = rot[6];	trans44_B[9] = rot[7];			trans44_B[10] = rot[8];		trans44_B[11] = 0; 
	trans44_B[12] = 0;		trans44_B[13] = 0;				trans44_B[14] = 0;			trans44_B[15] = 1; 

	float trans44_C[16];
	trans44_C[0] = rottar_normal_to_x_axis_T[0];	trans44_C[1] = rottar_normal_to_x_axis_T[1];			trans44_C[2] = rottar_normal_to_x_axis_T[2];		trans44_C[3] =  0; 
	trans44_C[4] = rottar_normal_to_x_axis_T[3];	trans44_C[5] = rottar_normal_to_x_axis_T[4];			trans44_C[6] = rottar_normal_to_x_axis_T[5];		trans44_C[7] =  0; 
	trans44_C[8] = rottar_normal_to_x_axis_T[6];	trans44_C[9] = rottar_normal_to_x_axis_T[7];			trans44_C[10] = rottar_normal_to_x_axis_T[8];		trans44_C[11] = 0; 
	trans44_C[12] = 0;								trans44_C[13] = 0;										trans44_C[14] = 0;									trans44_C[15] = 1; 

	float trans44_D[16];
	trans44_D[0] = 1;	trans44_D[1] = 0;			trans44_D[2] = 0;		trans44_D[3] = tslateFromOriginToTar.x; 
	trans44_D[4] = 0;	trans44_D[5] = 1;			trans44_D[6] = 0;		trans44_D[7] = tslateFromOriginToTar.y; 
	trans44_D[8] = 0;	trans44_D[9] = 0;			trans44_D[10] = 1;		trans44_D[11] = tslateFromOriginToTar.z; 
	trans44_D[12] = 0;	trans44_D[13] = 0;			trans44_D[14] = 0;		trans44_D[15] = 1; 

	float trans44BA[16];
	mul44(trans44_B, trans44_A, trans44BA);

	float trans44CBA[16];
	mul44(trans44_C, trans44BA, trans44CBA);

	float trans44[16];
	mul44(trans44_D, trans44CBA, trans44);

	/************out *******************/
	for (int i = 0; i < 16 ; ++i)
	{
		pPosesGPU[IdxInBatch * 16 + i] = trans44[i];
	}
}


int ComputeTransformForCorrespondencesCU(
		float* pPointModels,		// size: numOfPointModel * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		float* pPointModelsNormal,   // size: numOfPointModel * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		int numOfPointModel,
		float* pPointScene,			// size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		float* pPointSceneNormal,   // size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
		int numOfPoinScene,
		int *pSrcId0,				// size: batchSize
		int *pSrcId1,				// size: batchSize
		int *pTarId0,				// size: batchSize
		int *pTarId1,				// size: batchSize
		int batchSize,						
		float* pPosesGPU     // size: batchSize * 16
	)
{
	unsigned int threadsPerBlock = 1024;
	unsigned int totalThreadSize = batchSize;
	unsigned int blocksPerGrid = (totalThreadSize + threadsPerBlock - 1) / threadsPerBlock;
	ComputeTransformForCorrespondencesKernel << <blocksPerGrid, threadsPerBlock >> > (
		pPointModels,	
		pPointModelsNormal,
		numOfPointModel,
		pPointScene,	
		pPointSceneNormal,
		numOfPoinScene,
		pSrcId0,				
		pSrcId1,		
		pTarId0,			
		pTarId1,		
		batchSize,						
		pPosesGPU);

    gpuErrchk( cudaPeekAtLastError());
	gpuErrchk(cudaThreadSynchronize());

	return 0;
}