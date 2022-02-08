#pragma once

#include <math.h>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <execution>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

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
	KNNImage(const Mat& inputImg, int _width, int _height) : width(_width), height(_height), Mat(_height, _width, CV_8UC1, Scalar(0)) {

        auto start = high_resolution_clock::now();

        PreProcess_Serial(inputImg);

        auto stop = high_resolution_clock::now();

        cout << "Serial time:" << duration_cast<microseconds>(stop - start).count() << endl;

        start = high_resolution_clock::now();

        PreProcess_Parallel(inputImg);

        stop = high_resolution_clock::now();

        cout << "Parallel time:" << duration_cast<microseconds>(stop - start).count() << endl;
	}

    void PreProcess_Serial(const Mat& img) {

        const double scaleX = (double)img.size().width / width;
        const double scaleY = (double)img.size().height / height;


        for (unsigned i = 0; i < height * width; i++) {

            int y = (i / width) * scaleY;
            int x = (i % width) * scaleX;

            Vec3b bgrpixle = img.at<Vec3b>(y, x);

            int value_b = bgrpixle[0];
            int value_g = bgrpixle[1];
            int value_r = bgrpixle[2];

            uchar gray_value = (uchar)(0.114 * value_b + 0.587 * value_g + 0.299 * value_r);

            this->data[i] = gray_value;
        }
    }

    void PreProcess_Parallel(const Mat& img) {

        const double scaleX = (double)img.size().width / width;
        const double scaleY = (double)img.size().height / height;

        #pragma omp parallel for
        for (unsigned i = 0; i < height * width; i++) {

            int y = (i / width) * scaleY;
            int x = (i % width) * scaleX;

            Vec3b bgrpixle = img.at<Vec3b>(y, x);

            int value_b = bgrpixle[0];
            int value_g = bgrpixle[1];
            int value_r = bgrpixle[2];

            uchar gray_value = (uchar)(0.114 * value_b + 0.587 * value_g + 0.299 * value_r);

            this->data[i] = gray_value;
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

