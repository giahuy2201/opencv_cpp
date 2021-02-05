#include <iostream>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

// support only 3x3 kernel for now
void myFilter2D(Mat &src, Mat &dst, Mat &krnl)
{
    const int nRows = src.rows;
    const int nCols = src.cols;
    const int nChannels = src.channels();
    dst.create(nRows, nCols, src.type());
    // skip first n rows
    for (int i = 1; i < nRows; i++)
    {
        // pointers for each row
        uchar *prev = src.ptr<uchar>(i - 1);
        uchar *curr = src.ptr<uchar>(i);
        uchar *next = src.ptr<uchar>(i + 1);
        uchar *res = dst.ptr<uchar>(i);

        for (int j = nChannels; j < (nCols - 1) * nChannels; j++)
        {
            int topleft = prev[j - nChannels];
            int top = prev[j];
            int topright = prev[j + nChannels];
            int midleft = curr[j - nChannels];
            int mid = curr[j];
            int midright = curr[j + nChannels];
            int botleft = next[j - nChannels];
            int bot = next[j];
            int botright = next[j + nChannels];
            // multiply the corresponding and take the sum
            *res = saturate_cast<uchar>(topleft * krnl.at<float>(0, 0) + top * krnl.at<float>(0, 1) + topright * krnl.at<float>(0, 2) + midleft * krnl.at<float>(1, 0) + mid * krnl.at<float>(1, 1) + midright * krnl.at<float>(1, 2) + botleft * krnl.at<float>(2, 0) + bot * krnl.at<float>(2, 1) + botright * krnl.at<float>(2, 2));
            res++;
        }
    }
}

Mat gaussianKernel(int size, float sigma)
{
    int ctr = size / 2;
    float sum = 0;
    Mat mat = Mat::zeros(Size(size, size), CV_32F);
    // by default, matrix is stored in 1d array
    float *ptr = mat.ptr<float>(0);
    for (int i = 0; i < size * size; i++)
    {
        // calculate kernel pixel value based on x,y
        int x = i % size;
        int y = i / size;
        float e_pow = -(pow(x - ctr, 2) + pow(y - ctr, 2) + .0) / (2 * pow(sigma, 2));
        float value = pow(M_E, e_pow) / (2 * M_PI * pow(sigma, 2));
        // assign to the kernel
        ptr[i] = value;
        sum += value;
    }
    cout << "sum: " << sum << endl;
    // normalize the kernel by divide each pixel by the sum
    // since we didn't move ptr, it always point to the head of the array
    for (int i = 0; i < size * size; i++)
    {
        *ptr++ = *ptr / sum;
    }
    return mat;
}

int main()
{
    Mat img = imread("pikrepo.com.jpg", IMREAD_GRAYSCALE);
    imshow("original", img);
    cout << "depth: " << img.depth() << ", channels: " << img.channels() << endl;

    Mat kernel = gaussianKernel(3, 2);
    cout << kernel << endl;
    Mat out;
    myFilter2D(img, out, kernel);
    imshow("blurred", out);

    waitKey(0);

    return 0;
}