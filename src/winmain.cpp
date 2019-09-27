#include <string>

extern int preprocess(std::string object_name);
extern int stocs_est(std::string scene_path, std::string object_name);

int main(int argc, char** argv)
{
	int mode = 1;
	std::string object_name = "024_bowl";
	std::string scenePath = "D:/ronaldwork/model_matching/examples/ycb";

	if (mode == 0)
	{
		preprocess(object_name);
	}
	else
	{
		stocs_est(scenePath, object_name);
	}
}