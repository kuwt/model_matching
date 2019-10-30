#include <string>

// alg parameters
extern const float ds_voxel_size = 0.003; // In m, sampling size
extern const float normal_radius = 0.005; // In m radius for calculating normal vector of a point, depends on point cloud density
extern const int ppf_tr_discretization = 5; // In mm, same as when create model. 
extern const int ppf_rot_discretization = 5; // degrees, same as when create model. 
extern const float validPairMinDist = 0.04; // depends on object size

extern const float lcp_distthreshold = 0.0015; // for Congruent Set Matching and LCP computation, depends on point cloud density
extern const int goodPairsMax = 200;

// input parameters
extern const std::string model_scale = "m";			// input model scale
extern const std::string scene_scale = "mm";		// input scene scale

// running parameters
extern const int batchSize = 10000;
extern const int debug_flag = 0;
