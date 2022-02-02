#pragma once
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <filesystem>
#include <tuple>

using namespace std;
using namespace cv;
using std::filesystem::directory_iterator;

class KNNClassifier
{

	const string IMAGES_FOLDER_PATH = "E:\\Documents\\WorkSpace\\CIS6007_Assessment\\images\\";

	vector<tuple<Mat, string>> images;

	mutex mu;

public:

	KNNClassifier() {
		LoadTrainData();
	}

	const int GetImageNum() { return images.size(); }

	void LoadTrainData() {

		for (const auto& folderName : directory_iterator(IMAGES_FOLDER_PATH + "train")) {

			for (const auto& imageName : directory_iterator(folderName.path())) {

				cout << imageName;
			}
		}
	}
};

