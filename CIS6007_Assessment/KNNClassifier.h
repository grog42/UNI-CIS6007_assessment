#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <tuple>
#include <future>
#include <execution>

#include "KNNImage.h"

using namespace std;
using namespace cv;
using std::filesystem::directory_iterator;

class KNNClassifier
{

	const string IMAGES_FOLDER_PATH = "E:\\Documents\\WorkSpace\\CIS6007_Assessment\\images\\";

	int imageWidth;
	int imageHeight;

	/*
	* Vector stores path of image and assosiated label
	*/
	vector<unique_ptr<KNNImage>> images;

	vector<int> indexList;

public:

	KNNClassifier(int _imageWidth, int _imageHeight) : imageWidth(_imageWidth), imageHeight(_imageHeight) {

		auto dir = directory_iterator(IMAGES_FOLDER_PATH + "train");
		
		auto LoadImage = [&](string imgPath, string imgLable) {
			Mat img = imread(imgPath);

			if (img.empty()) 
				throw "";
			

			return make_unique<KNNImage>(KNNImage(img, imgLable, imageWidth, imageHeight));
		};

		vector<future<unique_ptr<KNNImage>>> tasks;

		for (const auto& folderName : dir) {

			string folderPath = folderName.path().u8string();
			int trimLength = IMAGES_FOLDER_PATH.size() + 6;
			string imgLable = folderPath.erase(0, trimLength);

			for (const auto& imageName : directory_iterator(folderName)) {

				string imgPath = imageName.path().u8string();

				tasks.push_back(async(launch::async, LoadImage, imgPath, imgLable));
			}
		}

		for (int i = 0; i < tasks.size(); i++) {
			images.push_back(tasks[i].get());
			//Loads index to each image into a vector
			indexList.push_back(i);
		}
	}

	string Classify(const KNNImage& inputImg, int k) {

		vector<tuple<double, string>> distMap = vector<tuple<double, string>>(images.size());

		//For each stored image the distance to the inputImg is calculated
		//The distance and lable are then added to a vector storing distances associated with labels
		auto GetDistances = [&](int i) {
			distMap[i] = make_tuple(images[i]->Dist_To_S(inputImg), images[i]->lable);
		};

		for_each(execution::seq, this->indexList.begin(), this->indexList.end(), GetDistances);
		
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

