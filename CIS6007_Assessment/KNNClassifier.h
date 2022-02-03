#pragma once
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <filesystem>
#include <tuple>

#include "KNNImage.h"

using namespace std;
using namespace cv;
using std::filesystem::directory_iterator;

class KNNClassifier
{

	const string IMAGES_FOLDER_PATH = "E:\\Documents\\WorkSpace\\CIS6007_Assessment\\images\\";

	/*
	* Vector stores path of image and assosiated label
	*/
	vector<tuple<KNNImage, string>> images;

	mutex mu;

public:

	KNNClassifier() {
		LoadTrainData();
	}

	void LoadTrainData() {

		for (const auto& folderName : directory_iterator(IMAGES_FOLDER_PATH + "train")) {

			string folderPath = folderName.path().u8string();
			int trimLength = IMAGES_FOLDER_PATH.size() + 6;
			string imgLable = folderPath.erase(0, trimLength);

			for (const auto& imageName : directory_iterator(folderName.path())) {

				const string imgPath = imageName.path().u8string();

				cout << imgLable << endl;

				images.push_back(tuple<KNNImage, string>(imread(imgPath), imgLable));
			}
		}
	}
};

