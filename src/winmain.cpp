#include <string>

extern int preprocess(std::string object_name);
extern int stocs_est(std::string scene_path, std::string object_name);
extern int testBruteforceReg(std::string scene_path, std::string object_name);
extern int gpucs(std::string scene_path, std::string object_name);
extern int gpucs2(std::string scene_path, std::string object_name);
extern int gpucs3(std::string scene_path, std::string object_path, std::string ppf_path);
extern int preprocess2(std::string model_path);

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

int main(int argc, char** argv)
{
	int mode = 1;

	if (mode == 0)
	{
		std::string modelpath = "D:/ronaldwork/model_matching/customtestcases/1/tmodel.ply";
		preprocess2(modelpath);
	}
	else
	{
		std::string modelpath = "./model_search.ply";
		std::string ppfpath = "./ppf_map";
		std::string scenePath = "D:/ronaldwork/model_matching/customtestcases/1/";
		gpucs3(scenePath, modelpath, ppfpath);
	}
}