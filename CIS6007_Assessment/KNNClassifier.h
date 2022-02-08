#pragma once
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <filesystem>
#include <tuple>
#include <stack>

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
	vector<tuple<KNNImage, string>> images;

	mutex mu;

public:

	KNNClassifier(int _imageWidth, int _imageHeight) : imageWidth(_imageWidth), imageHeight(_imageHeight) {
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

				Mat img = imread(imgPath);

				if (img.empty()) {

					cout << "Image not found" << endl;
					continue;
				}

				images.push_back(tuple<KNNImage, string>(KNNImage(img, imageWidth, imageHeight), imgLable));
			}
		}
	}

	string Classify(const Mat& inputImg, int k) {

		KNNImage img(inputImg, imageWidth, imageHeight);

		vector<tuple<double, string>> distMap = vector<tuple<double, string>>(images.size());

		//For each stored <image, label> tuple the distance to the inputed image is calculated
		//The distance and lable are then added to a vector storing distances associated with labels
		for (int i = 0; i < images.size(); i++)
			distMap[i] = make_tuple(KNNImage::Dist(get<0>(images[i]), img), get<1>(images[i]));
		

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
};

