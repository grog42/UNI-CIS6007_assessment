
#include <iostream>
#include <opencv2/opencv.hpp>

#include "KNNClassifier.h"

using namespace std;
using namespace cv;

Mat convert_to_grayscale(const Mat& rgb) {

    Mat gray_image(rgb.size().height, rgb.size().width, CV_8UC1, Scalar(0));

    for (int y = 0; y < rgb.size().width; y++) {
        for (int x = 0; x < rgb.size().height; x++) {

            Vec3b bgrpixle = rgb.at<Vec3b>(x, y);

            uchar gray_value = (uchar)(0.114 * bgrpixle[0] + 0.587 * bgrpixle[1] + 0.299 * bgrpixle[2]);

            gray_image.at<uchar>(x, y) = gray_value;
        }
    }

    return gray_image;
}

void convert_to_grauscale_serial(unsigned char* input, unsigned char* output, int start, int end, int nchannels) {

    auto j = start;
    auto number_of_pixles = end;

    for (auto i = 0; i < number_of_pixles; i += nchannels) {
        int value_b = input[0];
        int value_g = input[i + 1];
        int value_r = input[i + 2];

        output[j++] = (int)(0.114 * value_b + 0.587 * value_g + 0.299 * value_r);
    }
}

int main(int argc, char** argv)
{
    KNNClassifier classifier = KNNClassifier();

    classifier.GetImageNum();
}