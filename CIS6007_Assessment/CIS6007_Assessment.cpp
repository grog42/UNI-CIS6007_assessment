
#include <iostream>
#include <opencv2/opencv.hpp>

#include "KNNClassifier.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    KNNClassifier classifier = KNNClassifier(2000, 2000);

    Mat t1 = imread("E:\\Documents\\WorkSpace\\CIS6007_Assessment\\images\\test\\apple\\Image_1.jpg");
    Mat t2 = imread("E:\\Documents\\WorkSpace\\CIS6007_Assessment\\images\\test\\apple\\Image_2.jpg");

    KNNImage testImage1(t1, "test", 700, 700);
    KNNImage testImage2(t2, "test", 700, 700);



    string lable = classifier.Classify(KNNImage(t1, "apple", 2000, 2000), 10);

    cout << "Class is:" << lable << endl;

    waitKey(0); 

    return 0;
}