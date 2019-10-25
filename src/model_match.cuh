#pragma once


int  TransformPointsCU(
	float* pPointModels,		// size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointPerModel,
	int batchSize,
    float* pPoses,				// size: batchSize * 16 ;; t1[16], t2[16], ... 					
	float* pPointModelTranses     // size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	);

int  verifyPointsNearCU(
	float* pdists,        // size: numOfPointPerModel * batchSize ;;  t1 pt1 dist, t1 pt2 dist, ...  , t1 ptn dist, t2 pt1 dist, t2 pt2 dist, ...... 
	int numOfPointPerModel,
	int batchSize,
	float distThd,
    int* ppointsValids     // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	);

int  findAndVerifyNearestPointsCU(
	float* pPointModels,		// size: numOfPointPerModel * batchSize * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointPerModel,
	float *pPointScene,			// size: numOfPointScene * 3 ;; p1_x,P1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z ...
	int numOfPointScene,
	int batchSize,
	float distThd,
    int* ppointsValids     // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	);

int  computePoseLCP_CU(
	int* ppointsValids,   // size: numOfPointPerModel * batchSize ;; t1 pt1 valid, t1 pt2 valid, ...  ,t1 ptn valid , t2 pt1 valid, t2 pt2 valid, ...  ,t2 ptn valid , ...... 
	int numOfPointPerModel,
	int batchSize,
	float * pLCPs      // size: batchSize, LCP1, LCP2, ... LCPn	
	);



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
	);