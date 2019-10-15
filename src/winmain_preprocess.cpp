#include <stocs.hpp>

static std::string repo_path = "D:/ronaldwork/model_matching";

// All values in m
static float voxel_size = 0.01;		// sampling size
static float normal_radius = 0.005; // radius for calculating normal vector of a point
static float model_scale = 1.0;		// input model scale

// All values in mm
static int ppf_tr_discretization = 5;
static int ppf_rot_discretization = 5;

int preprocess(std::string object_name)
{
	std::string model_path = repo_path + "/models/" + object_name;

	stocs::pre_process_model(model_path + "/textured_vertices.ply",
		normal_radius,
		model_scale,
		1.0f,
		voxel_size,
		ppf_tr_discretization,
		ppf_rot_discretization,
		model_path + "/model_search.ply",
		model_path + "/ppf_map");

	return 0;
}