#include "rgbd.hpp"

#include <pcl/filters/random_sample.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/extract_indices.h>

// input parameters
extern const std::string model_scale;

// alg parameters
extern const float ds_voxel_size;
extern const float normal_radius;
extern const int ppf_tr_discretization;
extern const int ppf_rot_discretization;
extern const float validPairMinDist;


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
	std::map<std::vector<int>, std::vector<std::pair<int, int> > > ppf_map;

	/********** compute normal ********************/
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::io::loadPLYFile(src_model_location, *cloud);
	rgbd::compute_normal_pcl(cloud, normal_radius);

	//adding a negative sign. Reference frame is inside the object, so by default normals face inside.
	for (int i = 0; i < cloud->points.size(); i++)
	{
		cloud->points[i].normal[0] = -cloud->points[i].normal[0];
		cloud->points[i].normal[1] = -cloud->points[i].normal[1];
		cloud->points[i].normal[2] = -cloud->points[i].normal[2];
	}

	/************Downsample input and target point cloud ***************/
	/*
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_Src_DN(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	int downsampleIndexNumber = 1000;
	{
		pcl::RandomSample<pcl::PointXYZRGBNormal> Pcl_RandomSample;

		Pcl_RandomSample.setInputCloud(cloud);
		Pcl_RandomSample.setSample(downsampleIndexNumber);
		pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_Src_Dsample;
		Pcl_RandomSample.filter(cloud_Src_Dsample);
		*cloud_Src_DN = cloud_Src_Dsample;
	}
	*/
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
			Point3D p1 = point3d_sampled[id1];
			Point3D p2 = point3d_sampled[id2];

			float dist = std::sqrt(
				(p1.x() - p2.x()) * (p1.x() - p2.x())
				+ (p1.y() - p2.y()) * (p1.y() - p2.y())
				+ (p1.z() - p2.z()) * (p1.z() - p2.z()));

			if (dist < validPairMinDist)
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
		ds_voxel_size,
		ppf_tr_discretization,
		ppf_rot_discretization,
		"./model_search.ply",
		"./ppf_map");

	return 0;
}