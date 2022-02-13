#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <tuple>
#include <future>
#include <execution>
#include <math.h>

#include "KNNImage.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

class KNNClassifier
{
	const string dirPath;

	int imageWidth;
	int imageHeight;

	/*
	* Vector stores path of image and assosiated label
	*/
	vector<unique_ptr<KNNImage>> images;

	vector<int> indexList;

public:

	KNNClassifier(int _imageWidth, int _imageHeight, string _dirPath) : imageWidth(_imageWidth), imageHeight(_imageHeight), dirPath(_dirPath) {

		string trainDirPath = dirPath + "train";

		if (!fs::exists(trainDirPath)) {
			cout << "Dir not found" << endl;
			throw "";
		}
		
		auto LoadImage = [&](string imgPath, string imgLable) {
			Mat img = imread(imgPath);

			if (img.empty()) 
				throw "";
			
			return make_unique<KNNImage>(KNNImage(img, imgLable, imageWidth, imageHeight));
		};

		vector<future<unique_ptr<KNNImage>>> tasks;

		for (const auto& folderName : fs::directory_iterator(trainDirPath)) {

			string folderPath = folderName.path().u8string();
			int trimLength = dirPath.size() + 6;
			string imgLable = folderPath.erase(0, trimLength);

			for (const auto& imageName : fs::directory_iterator(folderName)) {

				string imgPath = imageName.path().u8string();

				if (!fs::exists(imgPath)) {
					cout << "Image not found" << endl;
					continue;
				}

				tasks.push_back(async(launch::async, LoadImage, imgPath, imgLable));
			}
		}

		for (int i = 0; i < tasks.size(); i++) {
			images.push_back(tasks[i].get());
			//Loads index to each image into a vector
			indexList.push_back(i);
		}
	}

	string Classify(const Mat& _inputImg, int k) {

		KNNImage inputImg(_inputImg, "", imageWidth, imageHeight);

		vector<tuple<double, string>> distMap = vector<tuple<double, string>>(images.size());

		//For each stored image the distance to the inputImg is calculated
		//The distance and lable are then added to a vector storing distances associated with labels
		const int depthNum = (log(images.size()) / log(2)) - 1;

		function<void(int, int, int)> Partition = [&](int start, int end, int depth) {

			if (depth > depthNum || depth == -1) {

				for(int i = start; i < end; i++) {
					distMap[i] = make_tuple(images[i]->Dist_To_P(inputImg), images[i]->lable);
				}
			}
			else {
				int mid = (start + end) / 2;

				auto f1 = async(launch::async, Partition, start, mid, ++depth);
				auto f2 = async(launch::async, Partition, mid, end, ++depth);

				f1.wait();
				f2.wait();
			}
		};

		Partition(0, this->images.size(), 0);
		
		//Tuples are sorted into asseding order based on distance
		//The sort function automaticaly picks the first element of the tuple to order
		sort(distMap.begin(), distMap.end());

		vector<string> closeLabes(k);

		//The k closest labels are added to an array
		for (int i = 0; i < k; i++) 
			closeLabes[i] = get<1>(distMap[i]);
		
		int bestQuant = 0;
		string bestLabel = closeLabes[0];

		//The number of times each label appears in the array is counted, the most common label is logged
		for (const string& label : closeLabes) {

			int quant = count(closeLabes.begin(), closeLabes.end(), label);

			if (bestQuant < quant) {
				bestQuant = quant;
				bestLabel = label;
			}
		}

		return bestLabel;
	}

	~KNNClassifier() {
		
	}
};

