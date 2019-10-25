#include "rgbd.hpp"

// All values in m
static float voxel_size = 0.005;		// sampling size
static float normal_radius = 0.005; // radius for calculating normal vector of a point
static std::string model_scale = "m";		// input model scale

// All values in mm
static int ppf_tr_discretization = 5;
static int ppf_rot_discretization = 5;


// supposed to be used offline. Otherwise this can be optimized by reading directly into point3d.
void  pre_process_model(std::string src_model_location,
	float normal_radius,
	float read_depth_scale,
	float write_depth_scale,
	float voxel_size,
	float ppf_tr_discretization,
	float ppf_rot_discretization,
	std::string dst_model_location,
	std::string dst_ppf_map_location)
{
	std::vector<Point3D> point3d, point3d_sampled;
	rgbd::UniformDistSampler sampler;
	std::map<std::vector<int>, std::vector<std::pair<int, int> > > ppf_map;

	/********** compute normal ********************/
	PCLPointCloud::Ptr cloud(new PCLPointCloud);
	pcl::io::loadPLYFile(src_model_location, *cloud);
	rgbd::compute_normal_pcl(cloud, normal_radius);

	//adding a negative sign. Reference frame is inside the object, so by default normals face inside.
	for (int i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].normal[0] = -cloud->points[i].normal[0];
		cloud->points[i].normal[1] = -cloud->points[i].normal[1];
		cloud->points[i].normal[2] = -cloud->points[i].normal[2];
	}

	/********** downsampling ********************/
	pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(voxel_size, voxel_size, voxel_size);
	sor.filter(*cloud);

	/********** convert from pcl format to custom format ********************/
	rgbd::load_ply_model(cloud, point3d_sampled, read_depth_scale);
	std::cout << "After sampling |M|= " << point3d_sampled.size() << std::endl;

	/********** compute ppf pairs ********************/
	float max_distance = 0;
	for (int id1 = 0; id1 < point3d_sampled.size(); id1++)
	{
		for (int id2 = 0; id2 < point3d_sampled.size(); id2++)
		{
			if (id1 == id2)
			{
				continue;
			}
			std::vector<int> ppf_;
			rgbd::ppf_compute(point3d_sampled[id1], point3d_sampled[id2], ppf_tr_discretization, ppf_rot_discretization, ppf_);
			rgbd::ppf_map_insert(ppf_map, ppf_, ppf_tr_discretization, ppf_rot_discretization, std::make_pair(id1, id2));

			float d = (point3d_sampled[id1].pos() - point3d_sampled[id2].pos()).norm();
			if (d > max_distance)
			{
				max_distance = d;
			}
		}
	}

	std::cout << "max distance is: " << max_distance << std::endl;

	std::cout << "saving ppf map..." << std::endl;
	rgbd::save_ppf_map(dst_ppf_map_location, ppf_map);
	rgbd::save_as_ply(dst_model_location, point3d_sampled, write_depth_scale);
}

int preprocess2(std::string model_path)
{
	float inscale = 1.0;
	if (model_scale == "m")
	{
		inscale = 1.0;
	}
	else if (model_scale == "mm")
	{
		inscale = 1.0 / 1000.0;
	}
	pre_process_model(model_path,
		normal_radius,
		inscale,
		1.0f,
		voxel_size,
		ppf_tr_discretization,
		ppf_rot_discretization,
		"./model_search.ply",
		"./ppf_map");

	return 0;
}