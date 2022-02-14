#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <future>
#include <execution>
#include <math.h>
#include <map>

#include "KNNImage.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

class KNNClassifier
{
	//The width and height which images must be projected to for the distance algorythm to work
	int imageWidth;
	int imageHeight;

	//Collection of images used to classify input images
	vector<unique_ptr<KNNImage>> images;

	bool parallelMode;

public:

	KNNClassifier(int _imageWidth, int _imageHeight, const string dirPath, bool _parallelMode = true) : imageWidth(_imageWidth), imageHeight(_imageHeight), parallelMode(_parallelMode) {

		string trainDirPath = dirPath + "train";

		if (!fs::exists(trainDirPath)) {
			cout << "Dir not found" << endl;
			return;
		}

		//Stores tasks waiting to be rejoined
		vector<future<unique_ptr<KNNImage>>> tasks;

		launch mode = (parallelMode) ? launch::async : launch::deferred;

		for (const auto& folderName : fs::directory_iterator(trainDirPath)) {

			string folderPath = folderName.path().u8string();
			const string imgLable = folderPath.erase(0, trainDirPath.size());

			for (const auto& imageName : fs::directory_iterator(folderName)) {

				string imgPath = imageName.path().u8string();

				if (!fs::exists(imgPath)) {
					cout << "Image not found" << endl;
					continue;
				}

				//Forks off task to be rejoined later on
				tasks.push_back(async(mode, [&](string imgPath, string imgLable) {
						return make_unique<KNNImage>(KNNImage(imread(imgPath), imgLable, imageWidth, imageHeight, parallelMode));
					}, imgPath, imgLable));
			}
		}

		//Rejoin tasks and push image to images
		for (int i = 0; i < tasks.size(); i++) 
			images.push_back(tasks[i].get());
		
	}

	pair<string, float> Classify(const Mat& _inputImg, int k) {

		KNNImage inputImg(_inputImg, "", imageWidth, imageHeight, parallelMode);

		//Create a vector which contatins the k closest labels
		vector<pair<double, string>> distMap = (parallelMode) ? GetDistances_par(inputImg, k) : GetDistances_ser(inputImg, k);

		//Create a Map which stores the quanitity of the k closest labels 
		map<string, int> labelCounts;

		for (int i = 0; i < k; i++) {
			
			string label = distMap[i].second;

			//If the label is not in the map it is added, else the count is incremented
			if (labelCounts.count(label) == 0)
				labelCounts.insert({ label, 1 });
			else
				labelCounts[label]++;
		}
		
		int bestQuant = 0;
		string bestLabel = "";

		//For loop finds the label with the higest quantity
		for (const pair<string, int>& pair : labelCounts) {

			if (bestQuant < pair.second) {
				bestQuant = pair.second;
				bestLabel = pair.first;
			}
		}

		float confidence = 100 * (labelCounts[bestLabel] / (float)k);

		return { bestLabel, confidence };
	}

	~KNNClassifier() {

	}

private:
	//Iterates serialy through each image and calculates its distance
	vector<pair<double, string>> GetDistances_ser(const KNNImage& inputImg, int k) {

		vector<pair<double, string>> dMap(images.size());

		for (int i = 0; i < dMap.size(); i++)
			dMap[i] = { images[i]->DistTo(inputImg, false), images[i]->lable };

		SortAndTrim(dMap, k);

		return dMap;
	}

	/*
	* Iterates through each image using an implementation of the divide and conquer parallel pattern
	* K is used to set the max lenth of the array to return
	*/
	vector<pair<double, string>> GetDistances_par(const KNNImage& inputImg, int k) {

		//Max number of times the number of images can be divided by 2
		const int maxDepth = log(images.size()) / log(2);

		/*After testing diffrent depths if was found that a depth of 4 offerd the highest level of speed up over the serial implmentation
		This number seems to match the number of threads the system can allocate as 2^4 = 16 and the system has 16 threads*/
		int targetDepth = 4;

		//Prevents start - end range going below 1
		if (targetDepth > maxDepth) {
			targetDepth = maxDepth;
		}

		//Breaks down problem and distributes to new threads for asynchronous operation
		function<vector<pair<double, string>>(int, int, int)> Partition = [&](int start, int end, int depth) {

			if (depth <= 0) {

				vector<pair<double, string>> dMap(end - start);

				for (int i = start; i < end; i++)
					dMap[i - start] = { images[i]->DistTo(inputImg, true), images[i]->lable };

				return dMap;
			}
			else {
				int mid = (start + end) / 2;

				auto f1 = async(launch::async, Partition, start, mid, depth - 1);
				auto f2 = async(launch::async, Partition, mid, end, depth - 1);

				auto v1 = f1.get();
				auto v2 = f2.get();

				v1.insert(v1.begin(), v2.begin(), v2.end());

				SortAndTrim(v1, k);

				return v1;
			}
		};

		return Partition(0, this->images.size(), targetDepth);
	}

	/*
	* v: Vector to be manipulated
	* maxLength: the trimed size of the vector
	*/
	void SortAndTrim(vector<pair<double, string>>& v, int maxLength) {

		sort(v.begin(), v.end());

		if (v.size() > maxLength) {
			v.resize(maxLength);
		}
	}
};

