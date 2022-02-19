
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>

#include "KNNClassifier.h"

using namespace std;
using namespace cv;
using namespace std::chrono;
namespace fs = filesystem;

vector<string> LoadTestImages(string dirPath) {

	string trainDirPath = dirPath + "test";

	if (!fs::exists(trainDirPath)) 
        throw exception("Image directory not found");
	
    vector<string> testImages;

	for (const auto& imgPath : fs::directory_iterator(trainDirPath)) {

        string path = imgPath.path().u8string();

        if (!fs::exists(path))
            throw exception("Image directory not found");

        testImages.push_back(path);
	}

    return testImages;
}

void TestSer(string dirPath, vector<string> testSet, int k, int imgResWidth, int imgResHeight) {

    auto start = high_resolution_clock::now();
    KNNClassifier classifier = KNNClassifier(imgResWidth, imgResHeight, dirPath, false);
    auto stop = high_resolution_clock::now();

    cout << "Serial Image loading/pre-processing time:" << duration_cast<microseconds>(stop - start).count() << endl;

    for (const auto& path : testSet) {

        start = high_resolution_clock::now();

        pair<string, float> result = classifier.Classify(imread(path), k);

        cout << "Class is:" << result.first << endl;
        cout << "Confidence is:" << result.second << endl;

        stop = high_resolution_clock::now();
        cout << "Serial classification time:" << duration_cast<microseconds>(stop - start).count() << endl;
    }
    
}

void TestPar(string dirPath, vector<string> testSet, int k, int imgResWidth, int imgResHeight) {

    auto start = high_resolution_clock::now();
    KNNClassifier classifier = KNNClassifier(imgResWidth, imgResHeight, dirPath, true);
    auto stop = high_resolution_clock::now();

    cout << "Parallel Image loading/pre-processing time:" << duration_cast<microseconds>(stop - start).count() << endl;

    for (const auto& path : testSet) {

        start = high_resolution_clock::now();

        pair<string, float> result = classifier.Classify(imread(path), k);

        cout << "Class is:" << result.first << endl;
        cout << "Confidence is:" << result.second << endl;

        stop = high_resolution_clock::now();
        cout << "Parallel classification time:" << duration_cast<microseconds>(stop - start).count() << endl;
    }
}

int main(int argc, char** argv)
{
    try {
        string dirPath = argv[1];
        int imgResWidth = stoi(argv[2]);
        int imgResHeight = stoi(argv[3]);

        string input;
        vector<string> testImages;

        if (!fs::exists(dirPath)) {
            throw exception("Image directory not found");
        }

        cout << "Enter path to test image or leave blank for the default set" << endl;
        getline(cin, input);

        if (input == "") {
            testImages = LoadTestImages(dirPath);
        }        
        else {
            if (!fs::exists(input))
                throw exception("Image not found");

            testImages.push_back(input);
        }

        cout << "Enter a value for K" << endl;
        getline(cin, input);

        int k = stoi(input);

        if (k < 1) {
            throw exception("K cannot be less than 1");
        }

        //TestPar(dirPath, testImages, k, imgResWidth, imgResHeight);

        TestSer(dirPath, testImages, k, imgResWidth, imgResHeight);
    }
    catch (exception& e) {
        cout << e.what() << endl;
        cout << "Program terminated" << endl;
    }
}