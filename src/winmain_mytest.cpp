#include <filesystem>
#include "rgbd.hpp"
#include "accelerators/kdtree.h"

#include <cuda.h>

#define FLANN_USE_CUDA
#include <flann/flann.hpp>

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


float compute_alignment_score_for_rigid_transform(
	std::vector<Point3D> &point3d_model,
	std::vector<Point3D>& point3d_scene,
	const Eigen::Ref<const Eigen::Matrix<float, 4, 4>>& mat,
	Super4PCS::KdTree<float> &kd_tree_)
{
	// We allow factor 2 scaling in the normalization.
	const float epsilon = distance_threshold;
	float weighted_match = 0;

	const size_t number_of_points_model = point3d_model.size();
	const float sq_eps = epsilon * epsilon;

	for (int i = 0; i < number_of_points_model; ++i)
	{
		// Use the kdtree to get the nearest neighbor
		Super4PCS::KdTree<float>::Index resId =
			kd_tree_.doQueryRestrictedClosestIndex(
			(mat * point3d_model[i].pos().homogeneous()).head<3>(),
				sq_eps);

		if (resId != Super4PCS::KdTree<float>::invalidIndex())
		{
			Point3D::VectorType n_q = mat.block<3, 3>(0, 0) * point3d_model[i].normal();

			float angle_n = std::acos(point3d_scene[resId].normal().dot(n_q)) * 180 / M_PI;

			// angle_n = std::min(angle_n, float(fabs(180-angle_n)));

			if (angle_n < 30)
			{
				weighted_match += point3d_scene[resId].class_probability();
			}
		}
	}
	return weighted_match / float(number_of_points_model);
}

int testBruteforceReg(std::string scene_path, std::string object_name)
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
				//e.srcCoord[0] = p1;
				//e.srcCoord[1] = p2;
				//e.tarCoord[0] = point3d_model[e.tarId[0]];
				//e.tarCoord[1] = point3d_model[e.tarId[1]];
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
	int verifyMax = 1000000000;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < poseEsts.size() && i < verifyMax; ++i)
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
		// debug
		if (0)
		{
			Point3D::VectorType src_p1_debug = src_p1;
			Point3D::VectorType src_p1_to_p2 = (src_p2 - src_p1);
			src_p1_to_p2.normalize();
			Point3D::VectorType src_p2_debug = src_p1 + src_p1_to_p2;
			Point3D::VectorType src_p3_debug = src_p1 + src_n1;

			Point3D::VectorType tar_p1_debug = tar_p1;
			Point3D::VectorType tar_p1_to_p2 = (tar_p2 - tar_p1);
			tar_p1_to_p2.normalize();
			Point3D::VectorType tar_p2_debug = tar_p1 + tar_p1_to_p2;
			Point3D::VectorType tar_p3_debug = tar_p1 + tar_n1;

			{
				std::string path;
				path = debug_path + "/srcPts.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n",src_p1_debug.x(), src_p1_debug.y(),src_p1_debug.z());
				std::fprintf(f, "v %f %f %f \n", src_p2_debug.x(), src_p2_debug.y(), src_p2_debug.z());
				std::fprintf(f, "v %f %f %f \n", src_p3_debug.x(), src_p3_debug.y(), src_p3_debug.z());
				
				std::fclose(f);
			}

			{
				std::string path;
				path = debug_path + "/tarPts.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", tar_p1_debug.x(), tar_p1_debug.y(), tar_p1_debug.z());
				std::fprintf(f, "v %f %f %f \n", tar_p2_debug.x(), tar_p2_debug.y(), tar_p2_debug.z());
				std::fprintf(f, "v %f %f %f \n", tar_p3_debug.x(), tar_p3_debug.y(), tar_p3_debug.z());

				std::fclose(f);
			}

			Point3D::VectorType src_p1_debug_trans;
			Point3D::VectorType src_p2_debug_trans;
			Point3D::VectorType src_p3_debug_trans;

			src_p1_debug_trans = src_p1_debug + tslateFromSrcToOrigin;
			src_p2_debug_trans = src_p2_debug + tslateFromSrcToOrigin;
			src_p3_debug_trans = src_p3_debug + tslateFromSrcToOrigin;
			{
				std::string path;
				path = debug_path + "/srcPts_trans.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", src_p1_debug_trans.x(), src_p1_debug_trans.y(), src_p1_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p2_debug_trans.x(), src_p2_debug_trans.y(), src_p2_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p3_debug_trans.x(), src_p3_debug_trans.y(), src_p3_debug_trans.z());
				std::fclose(f);
			}
			src_p1_debug_trans = rotation * src_p1_debug_trans;
			src_p2_debug_trans = rotation * src_p2_debug_trans;
			src_p3_debug_trans = rotation * src_p3_debug_trans;
			{
				std::string path;
				path = debug_path + "/srcPts_trans2.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", src_p1_debug_trans.x(), src_p1_debug_trans.y(), src_p1_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p2_debug_trans.x(), src_p2_debug_trans.y(), src_p2_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p3_debug_trans.x(), src_p3_debug_trans.y(), src_p3_debug_trans.z());
				std::fclose(f);
			}
			src_p1_debug_trans = rotation_tar_normal_to_x_axis_transpose * src_p1_debug_trans;
			src_p2_debug_trans = rotation_tar_normal_to_x_axis_transpose * src_p2_debug_trans;
			src_p3_debug_trans = rotation_tar_normal_to_x_axis_transpose * src_p3_debug_trans;
			{
				std::string path;
				path = debug_path + "/srcPts_trans3.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", src_p1_debug_trans.x(), src_p1_debug_trans.y(), src_p1_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p2_debug_trans.x(), src_p2_debug_trans.y(), src_p2_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p3_debug_trans.x(), src_p3_debug_trans.y(), src_p3_debug_trans.z());
				std::fclose(f);
			}
			src_p1_debug_trans = tslateFromOriginToTar + src_p1_debug_trans;
			src_p2_debug_trans = tslateFromOriginToTar + src_p2_debug_trans;
			src_p3_debug_trans = tslateFromOriginToTar + src_p3_debug_trans;
			{
				std::string path;
				path = debug_path + "/srcPts_trans4.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", src_p1_debug_trans.x(), src_p1_debug_trans.y(), src_p1_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p2_debug_trans.x(), src_p2_debug_trans.y(), src_p2_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", src_p3_debug_trans.x(), src_p3_debug_trans.y(), src_p3_debug_trans.z());
				std::fclose(f);
			}

			Point3D::VectorType tar_p1_debug_trans;
			Point3D::VectorType tar_p2_debug_trans;
			Point3D::VectorType tar_p3_debug_trans;

			tar_p1_debug_trans = tar_p1_debug - tslateFromOriginToTar;
			tar_p2_debug_trans = tar_p2_debug - tslateFromOriginToTar;
			tar_p3_debug_trans = tar_p3_debug - tslateFromOriginToTar;
			{
				std::string path;
				path = debug_path + "/tarPts_trans.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", tar_p1_debug_trans.x(), tar_p1_debug_trans.y(), tar_p1_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", tar_p2_debug_trans.x(), tar_p2_debug_trans.y(), tar_p2_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", tar_p3_debug_trans.x(), tar_p3_debug_trans.y(), tar_p3_debug_trans.z());

				std::fclose(f);
			}
			tar_p1_debug_trans = rotation_tar_normal_to_x_axis * tar_p1_debug_trans;
			tar_p2_debug_trans = rotation_tar_normal_to_x_axis * tar_p2_debug_trans;
			tar_p3_debug_trans = rotation_tar_normal_to_x_axis * tar_p3_debug_trans;
			{
				std::string path;
				path = debug_path + "/tarPts_trans2.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", tar_p1_debug_trans.x(), tar_p1_debug_trans.y(), tar_p1_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", tar_p2_debug_trans.x(), tar_p2_debug_trans.y(), tar_p2_debug_trans.z());
				std::fprintf(f, "v %f %f %f \n", tar_p3_debug_trans.x(), tar_p3_debug_trans.y(), tar_p3_debug_trans.z());

				std::fclose(f);
			}

			Eigen::Vector4f src_p1_debug4, src_p2_debug4, src_p3_debug4;
			src_p1_debug4 << src_p1_debug.x(), src_p1_debug.y(), src_p1_debug.z(), 1.0;
			src_p2_debug4 << src_p2_debug.x(), src_p2_debug.y(), src_p2_debug.z(), 1.0;
			src_p3_debug4 << src_p3_debug.x(), src_p3_debug.y(), src_p3_debug.z(), 1.0;

			Eigen::Vector4f src_p1_debug_dirtrans4 = trans44 * src_p1_debug4;
			Eigen::Vector4f src_p2_debug_dirtrans4 = trans44 * src_p2_debug4;
			Eigen::Vector4f src_p3_debug_dirtrans4 = trans44 * src_p3_debug4;

			Point3D::VectorType src_p1_debug_dirtrans = src_p1_debug_dirtrans4.head<3>();
			Point3D::VectorType src_p2_debug_dirtrans = src_p2_debug_dirtrans4.head<3>();
			Point3D::VectorType src_p3_debug_dirtrans = src_p3_debug_dirtrans4.head<3>();
			{
				std::string path;
				path = debug_path + "/srcPts_trans_dir.obj";
				std::FILE* f = std::fopen(path.c_str(), "w");

				std::fprintf(f, "v %f %f %f \n", src_p1_debug_dirtrans.x(), src_p1_debug_dirtrans.y(), src_p1_debug_dirtrans.z());
				std::fprintf(f, "v %f %f %f \n", src_p2_debug_dirtrans.x(), src_p2_debug_dirtrans.y(), src_p2_debug_dirtrans.z());
				std::fprintf(f, "v %f %f %f \n", src_p3_debug_dirtrans.x(), src_p3_debug_dirtrans.y(), src_p3_debug_dirtrans.z());
				std::fclose(f);
			}
		}
	}
	finish = std::chrono::high_resolution_clock::now();
	std::cout << "calculate transform " << poseEsts.size() << " in "
		<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
		<< " milliseconds\n";
	/***********  verify pose  ********************/
	float best_lcp;
	int best_index;
#ifdef CPU
	{
		start = std::chrono::high_resolution_clock::now();
		// Build the kdtree.
		size_t number_of_points_scene = point3d_scene.size();
		Super4PCS::KdTree<float> kd_tree_ = Super4PCS::KdTree<float>(number_of_points_scene);

		for (size_t i = 0; i < number_of_points_scene; ++i)
		{
			kd_tree_.add(point3d_scene[i].pos());
		}
		kd_tree_.finalize();
		finish = std::chrono::high_resolution_clock::now();
		std::cout << "build KD tree for scene " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";

		start = std::chrono::high_resolution_clock::now();
		std::cout << "Transforms to verify: " << poseEsts.size() << std::endl;
		float max_score = 0;
		int index = -1;

		for (int i = 0; i < poseEsts.size() && i < verifyMax; i++)
		{
			float lcp = compute_alignment_score_for_rigid_transform(
						point3d_model,
						point3d_scene,
						poseEsts[i].trans44,
						kd_tree_ );
			poseEsts[i].lcp = lcp;

			if (lcp > max_score)
			{
				max_score = lcp;
				index = i;
			}
		}
		best_lcp = max_score;
		best_index = index;
		std::cout << "best index: " << best_index << ", maximum score: " << best_lcp << std::endl;

		finish = std::chrono::high_resolution_clock::now();
		std::cout << "verify transform " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";
	}
#endif

	{
		start = std::chrono::high_resolution_clock::now();
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

		// KNN, build kd-tree
		flann::Matrix<float> dataSet(pPointScene, number_of_points_scene, 3, 3 * sizeof(float));
		flann::KDTreeCuda3dIndexParams   cudaParams(32);
		flann::KDTreeCuda3dIndex<::flann::L2<float>> KnnSearch(dataSet, cudaParams);
		KnnSearch.buildIndex();

		finish = std::chrono::high_resolution_clock::now();
		std::cout << "build KD tree for scene " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";

		int batchSize = 10;
		size_t number_of_points_model = point3d_model.size();
		float* pPointModel = new float[number_of_points_model * 3 * batchSize];
		float* pPointModelTrans = new float[number_of_points_model * 3 * batchSize];

		int numberOfBatch = poseEsts.size() / batchSize;
		for (int j = 0; j < batchSize; ++j)
		{
			for (int i = 0; i < number_of_points_model; i++)
			{
				pPointModel[j * number_of_points_model + i * 3 + 0] = point3d_model[i].x();
				pPointModel[j * number_of_points_model + i * 3 + 1] = point3d_model[i].y();
				pPointModel[j * number_of_points_model + i * 3 + 2] = point3d_model[i].z();
			}
		}

		::flann::Matrix<int> indices(new int[batchSize * number_of_points_model * 1], batchSize* number_of_points_model, 1);
		::flann::Matrix<float> dists(new float[batchSize * number_of_points_model * 1], batchSize* number_of_points_model, 1);
		::flann::SearchParams searchParam(8, 0, true);
		start = std::chrono::high_resolution_clock::now();
		for (int k = 0; k < numberOfBatch; ++k)
		{
			for (int j = 0; j < batchSize; ++j)
			{
				for (int i = 0; i < number_of_points_model; i++)
				{
					Eigen::Vector3f v = (poseEsts[k * batchSize + j].trans44 * point3d_model[i].pos().homogeneous()).head<3>();
					pPointModelTrans[j * number_of_points_model + i * 3 + 0] = v.x();
					pPointModelTrans[j * number_of_points_model + i * 3 + 1] = v.y();
					pPointModelTrans[j * number_of_points_model + i * 3 + 2] = v.z();
				}
			}

			// KNN, search neighbor
			flann::Matrix<float> modelSet(pPointModelTrans, 
				batchSize * number_of_points_model, 3, 3 * sizeof(float));
			KnnSearch.knnSearchGpu(modelSet, indices, dists, 1, searchParam);

		}

		delete[] pPointScene;
		delete[] pPointModel;
		delete[] pPointModelTrans;
		delete[] indices.ptr();
		delete[] dists.ptr();


		finish = std::chrono::high_resolution_clock::now();
		std::cout << "verify transform " << poseEsts.size() << " in "
			<< std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() * 0.001
			<< " milliseconds\n";
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