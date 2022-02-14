#pragma once

#include <math.h>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <execution>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

/*
* A class which adds functionalty to the opencv Mat class
* The added functions prepare the image for use in KNN classification
*/
class KNNImage : public Mat
{
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
	KNNImage(const Mat& inputImg, string _lable, int width, int height, bool parallelMode = true): lable(_lable), indexList(height* width), Mat(height, width, CV_8UC1, Scalar(0)) {

        //This is a range of shared resources which are pre loaded to save time during execution
        //const is used to ensure the resources remain safe when accessed in parallel
        const double scaleX = (double)inputImg.size().width / width;
        const double scaleY = (double)inputImg.size().height / height;
        const uchar* imgData = (uchar*)inputImg.data;
        const uint nChannels = inputImg.channels();
        const uint inputImgWidth = inputImg.cols;

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
        
        if (parallelMode) 
            for_each(execution::par, indexList.begin(), indexList.end(), ProcessPixle);
        else 
            for_each(execution::seq, indexList.begin(), indexList.end(), ProcessPixle);       
	}

    /*
    * Distance = sqrt(Sum((In-Jn)^2))
    * Where I is image one, J is image two and n represents the index of a pixle
    * Images must have the same width and height
    */
    double DistTo(const KNNImage& img, bool parallelMode) {

        //Check that images are compatible
        if (this->rows != img.rows ||
            this->cols != img.cols) throw invalid_argument("Images must be the same size");

        const uchar* data_a = this->data;
        const uchar* data_b = img.data;

        vector<double> distances(indexList.size());

        auto pixleDistance = [&](int i) {
            distances[i] = pow(data_a[i] - data_b[i], 2);
        };

        if (parallelMode) {
            for_each(execution::par, this->indexList.begin(), this->indexList.end(), pixleDistance);
            return sqrt(reduce(execution::par, distances.begin(), distances.end()));
        }

        for_each(execution::seq, this->indexList.begin(), this->indexList.end(), pixleDistance);
        return sqrt(reduce(execution::seq, distances.begin(), distances.end()));
    }

    ~KNNImage() {
        
    }
};

