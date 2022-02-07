#pragma once

#include <math.h>
#include <stdexcept>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
* A class which adds functionalty to the opencv Mat class
* The added functions prepare the image for use in KNN classification
*/
class KNNImage : public Mat
{
    int width;
    int height;

public:

    /*
    * If constructed using an exsisting image, the image is converted to grayscale and scaled
    * This pre processing of the image makes it usable by the KNN algorythm
    */
	KNNImage(const Mat& img, int _width, int _height) : width(_width), height(_height), Mat(_height, _width, CV_8UC1, Scalar(0)) {
		
        double newImgScaleX = width / (double)img.size().width;
        double newImgScaleY = height / (double)img.size().height;

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {

                Vec3b bgrpixle = img.at<Vec3b>(y, x);

                uchar gray_value = (uchar)(0.114 * bgrpixle[0] + 0.587 * bgrpixle[1] + 0.299 * bgrpixle[2]);

                int xd = (int)(x * newImgScaleX);
                int yd = (int)(y * newImgScaleY);

                if (xd >= width || yd >= height) {
                    int i = 0;
                }

                this->at<uchar>(yd, xd) = gray_value;
            }
        }
	}

    /*
    * Distance = sqrt(Sum((In-Jn)^2))
    * Where I is image one, J is image two and n represents the index of a pixle
    * Images must have the same width and height
    */
    static double Dist(const KNNImage& a, const KNNImage& b) {

        //Check that images are compatible
        if (a.height != b.height ||
            a.width != b.width) throw invalid_argument("Images must be the same size");

        double sum = 0;

        for (int y = 0; y < a.size().width; y++) {
            for (int x = 0; x < a.size().height; x++) {

                const uchar& pA = a.at<uchar>(x, y);
                const uchar& pB = b.at<uchar>(x, y);
                sum += pow(pA - pB, 2);
            }
        }

        return sqrt(sum);
    }
};

