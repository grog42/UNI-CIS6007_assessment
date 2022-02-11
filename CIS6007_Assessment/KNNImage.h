#pragma once

#include <math.h>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <execution>
#include <chrono>
#include <omp.h>
#include <execution>
using namespace std::execution;

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

    /* A vector listing the 1d index of each pixle
    *  This is used to parse a pixles index to for_each functions 
    *  The index is used as it conveys position data 
    */
    vector<int> indexList;

public:

    const string lable;

    /*
    * If constructed using an exsisting image, the image is converted to grayscale and scaled
    * This pre processing of the image makes it usable by the KNN algorythm
    */
	KNNImage(const Mat& inputImg, string lable, int _width, int _height): lable(lable), width(_width), height(_height), indexList(height* width), Mat(_height, _width, CV_8UC1, Scalar(0)) {

        /*This is a range of varables which are pre loaded to save time during execution*/
        const double scaleX = (double)inputImg.size().width / width;
        const double scaleY = (double)inputImg.size().height / height;
        const uchar* imgData = (uchar*)inputImg.data;
        const uint nChannels = inputImg.channels();
        const uint inputImgWidth = inputImg.size().width;

        //Loads index to each pixle into a vector
        for (int i = 0; i < indexList.size(); i++)
            indexList[i] = i;
 
        auto ProcessPixle = [&](int i) {

            //x and y vals are calculated diffrently as the double value must be cast into uint before being used to locate the ptr
            uint y = (i / width) * scaleY;
            uint x = (i % width) * scaleX;

            uint ptr = ((y * inputImgWidth) + x) * nChannels;

            uint value_b = imgData[ptr];
            uint value_g = imgData[ptr + 1];
            uint value_r = imgData[ptr + 2];

            this->data[i] = (uchar)(0.114 * value_b + 0.587 * value_g + 0.299 * value_r);
        };
        
        
        /*
        auto start = high_resolution_clock::now();
        for_each(execution::seq, indexList.begin(), indexList.end(), ProcessPixle);
        auto stop = high_resolution_clock::now();

        cout << "Serial time:" << duration_cast<microseconds>(stop - start).count() << endl;
        */

        auto start = high_resolution_clock::now();
        for_each(execution::par, indexList.begin(), indexList.end(), ProcessPixle);
        auto stop = high_resolution_clock::now();

        cout << "Parallel time:" << duration_cast<microseconds>(stop - start).count() << endl;      
	}

    /*
    * Distance = sqrt(Sum((In-Jn)^2))
    * Where I is image one, J is image two and n represents the index of a pixle
    * Images must have the same width and height
    */
    double Dist_To_S(const KNNImage& img) {

        //Check that images are compatible
        if (this->height != img.height ||
            this->width != img.width) throw invalid_argument("Images must be the same size");

        double sum = 0;

        const uchar* data_a = this->data;
        const uchar* data_b = img.data;

        vector<double> distances(indexList.size());

        auto pixleDistance = [&](int i) {
            distances[i] = pow(data_a[i] - data_b[i], 2);
        };

        for_each(execution::seq, this->indexList.begin(), this->indexList.end(), pixleDistance);

        auto acc_result = reduce(execution::seq, distances.begin(), distances.end());

        return sqrt(acc_result);
    }

    double Dist_To_P(const KNNImage& img) {

        //Check that images are compatible
        if (this->height != img.height ||
            this->width != img.width) throw invalid_argument("Images must be the same size");

        double sum = 0;

        const uchar* data_a = this->data;
        const uchar* data_b = img.data;

        vector<double> distances(indexList.size());

        auto pixleDistance = [&](int i) {
            distances[i] = pow(data_a[i] - data_b[i], 2);
        };

        for_each(execution::par, this->indexList.begin(), this->indexList.end(), pixleDistance);

        auto acc_result = reduce(execution::par, distances.begin(), distances.end());

        return sqrt(acc_result);
    }

    ~KNNImage() {
        ~Mat();
    }
};

