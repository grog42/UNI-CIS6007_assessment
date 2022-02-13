
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

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

    KNNImage testImage1(t1, "test", 700, 700);
    KNNImage testImage2(t2, "test", 700, 700);
        
    KNNClassifier classifier = KNNClassifier(2000, 2000, dirPath);

    string lable = classifier.Classify(t1, 10);

    cout << "Class is:" << lable << endl;

    return 0;
}