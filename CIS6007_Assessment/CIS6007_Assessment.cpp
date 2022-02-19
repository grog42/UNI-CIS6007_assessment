
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>

#include "KNNClassifier.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

const int testWidth = 200;
const int testHeight = 200;

vector<pair<string, string>> LoadTestImage(string dirPath) {

    if (!fs::exists(dirPath))
        throw exception("Image directory not found");

    string label;

    cout << "Please enter the images lable" << endl;
    cin >> label;

    return { {dirPath, label} };
}

vector<pair<string, string>> LoadTestImages(string dirPath) {

	string trainDirPath = dirPath + "test";

	if (!fs::exists(trainDirPath)) 
        throw exception("Image directory not found");
	
    vector<pair<string, string>> testImages;

	for (const auto& folderName : fs::directory_iterator(trainDirPath)) {

        //Use name of folder a image lable
		const string imgLable = folderName.path().u8string().erase(0, trainDirPath.size() + 1);

		for (const auto& imageName : fs::directory_iterator(folderName)) {

			string imgPath = imageName.path().u8string();
            testImages.push_back({ imgPath , imgLable});
		}
	}

    return testImages;
}

void TestSer(string dirPath, vector<pair<string, string>> testSet, int k) {

    auto start = high_resolution_clock::now();
    KNNClassifier classifier = KNNClassifier(testWidth, testHeight, dirPath, false);
    auto stop = high_resolution_clock::now();

    cout << "Serial Image loading/pre-processing time:" << duration_cast<microseconds>(stop - start).count() << endl;

    for (const auto& item : testSet) {

        start = high_resolution_clock::now();

        Mat img = imread(item.first);
        pair<string, float> result = classifier.Classify(img, k);
        cout << "Class is:" << result.first << "  Expected: " << item.second << endl;
        cout << "Confidence is:" << result.second << endl;

        stop = high_resolution_clock::now();
        cout << "Serial classification time:" << duration_cast<microseconds>(stop - start).count() << endl;
    }
    
}

void TestPar(string dirPath, vector<pair<string, string>> testSet, int k) {

    auto start = high_resolution_clock::now();
    KNNClassifier classifier = KNNClassifier(testWidth, testHeight, dirPath, true);
    auto stop = high_resolution_clock::now();

    cout << "Parallel Image loading/pre-processing time:" << duration_cast<microseconds>(stop - start).count() << endl;

    for (const auto& item : testSet) {

        start = high_resolution_clock::now();
        Mat img = imread(item.first);
        pair<string, float> result = classifier.Classify(img, k);
        cout << "Class is:" << result.first << "  Expected: " << item.second << endl;
        cout << "Confidence is:" << result.second << endl;

        stop = high_resolution_clock::now();
        cout << "Parallel classification time:" << duration_cast<microseconds>(stop - start).count() << endl;
    }
}

int main(int argc, char** argv)
{
    try {
        string dirPath = argv[1];

        if (!fs::exists(dirPath)) {
            throw exception("Image directory not found");
        }

        string input;

        cout << "Enter path to test image or leave blank for the default set" << endl;
        getline(cin, input);

        auto testImages = (input == "") ? LoadTestImages(dirPath) : LoadTestImage(input);

        cout << "Enter a value for K" << endl;
        getline(cin, input);

        int k = stoi(input);

        if (k < 1) {
            throw exception("K cannot be less than 1");
        }

        TestPar(dirPath, testImages, k);

        //TestSer(dirPath, testImages, k);
    }
    catch (exception& e) {
        cout << e.what() << endl;
        cout << "Program terminated" << endl;
    }
}