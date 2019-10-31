#include <string>


extern int preprocess2(std::string model_path);
extern int gpucs3(std::string scene_path, std::string object_path, std::string ppf_path);
extern int gpucs6(std::string scene_path, std::string object_path, std::string ppf_path);
extern int gpucs7(std::string scene_path, std::string object_path, std::string ppf_path);

extern int gpucs8_preprocess(std::string model_path);
extern int gpucs8(std::string scene_path, std::string object_path, std::string ppf_path);

extern void convetXYZIsValidMap(std::string scene_path, std::string object_name);

int main(int argc, char** argv)
{
	int mode = 1;

	if (mode == 0)
	{
		std::string modelpath = "D:/ronaldwork/model_matching/customtestcases/h1/hdensemodel.ply";
		gpucs8_preprocess(modelpath);
	}
	else if (mode == 1)
	{
		std::string modelpath = "./model_search.ply";
		std::string ppfpath = "./ppf_map";
		std::string scenePath = "D:/ronaldwork/model_matching/customtestcases/h1/";
		gpucs8(scenePath, modelpath, ppfpath);
	}
	else
	{
		std::string object_name = "024_bowl";
		std::string scenePath = "D:/ronaldwork/model_matching/examples/ycb";
		convetXYZIsValidMap(scenePath, object_name);
	}
}

/*
extern int preprocess(std::string object_name);
extern int stocs_est(std::string scene_path, std::string object_name);
extern int testBruteforceReg(std::string scene_path, std::string object_name);
extern int gpucs(std::string scene_path, std::string object_name);
extern int gpucs2(std::string scene_path, std::string object_name);


int oldapi()
{
	int mode = 4;
	std::string object_name = "024_bowl";
	std::string scenePath = "D:/ronaldwork/model_matching/examples/ycb";

	if (mode == 0)
	{
		preprocess(object_name);
	}
	else if (mode == 1)
	{
		stocs_est(scenePath, object_name);
	}
	else if (mode == 2)
	{
		testBruteforceReg(scenePath, object_name);
	}
	else if (mode == 3)
	{
		gpucs(scenePath, object_name);
	}
	else
	{
		gpucs2(scenePath, object_name);
	}

	return 0;
}
*/