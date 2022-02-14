
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "KNNClassifier.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

int main(int argc, char** argv)
{
    string dirPath = argv[1];

    if (!fs::exists(dirPath)) {
        cout << "Image directory not found" << endl;
        return 0;
    }

    Mat t1 = imread(dirPath + "test\\apple\\Image_1.jpg");
    Mat t2 = imread(dirPath + "test\\apple\\Image_2.jpg");

    auto start_par = high_resolution_clock::now();
    KNNClassifier classifier_par = KNNClassifier(2000, 2000, dirPath, true);
    pair<string, float> result_par = classifier_par.Classify(t1, 10);
    auto stop_par = high_resolution_clock::now();

    cout << "Parallel time:" << duration_cast<microseconds>(stop_par - start_par).count() << endl;
        
    auto start_ser = high_resolution_clock::now();
    KNNClassifier classifier_ser = KNNClassifier(2000, 2000, dirPath, false);
    pair<string, float> result_ser = classifier_par.Classify(t1, 10);
    auto stop_ser = high_resolution_clock::now();

    cout << "Serial time:" << duration_cast<microseconds>(stop_ser - start_ser).count() << endl;

    cout << "Class is:" << result_par.first << endl;
    cout << "Confidence is:" << result_par.second << endl;

    return 0;
}