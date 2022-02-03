#pragma once

#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
* A class which adds functionalty to the opencv Mat class
* The added functions prepare the image for use in KNN classification
*/
class KNNImage : public Mat
{

public:

    /*
    * If constructed using an exsisting image, the image is converted to grayscale 
    */
	KNNImage(const Mat& img) : Mat(img.size().height, img.size().width, CV_8UC1, Scalar(0)){
		
        for (int y = 0; y < img.size().width; y++) {
            for (int x = 0; x < img.size().height; x++) {

                Vec3b bgrpixle = img.at<Vec3b>(x, y);

                uchar gray_value = (uchar)(0.114 * bgrpixle[0] + 0.587 * bgrpixle[1] + 0.299 * bgrpixle[2]);

                this->at<uchar>(x, y) = gray_value;
            }
        }
	}

    /*
    * Distance = sqrt(Sum((In-Jn)^2))
    * Where I is image one, J is image two and n represents the index of a pixle
    * Images must have the same width and height
    */
    float DistTo(const Mat& img) {

        double sum = 0;

        for (int y = 0; y < this->size().width; y++) {
            for (int x = 0; x < this->size().height; x++) {

                const uchar& pA = this->at<uchar>(x, y);
                const uchar& pB = img.at<uchar>(x, y);
                sum += pow(pA - pB, 2);
            }
        }

        return sqrt(sum);
    }
};

