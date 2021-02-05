#include <iostream>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

// calculate Laplacian
Mat calcKernel(int size, float sigma)
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
        float LoG = pow(M_E, e_pow) * 4 * (pow(x - ctr, 2) + pow(y - ctr, 2)) / (2 * M_PI * pow(sigma, 2));
        // assign to the kernel
        ptr[i] = LoG;
        sum += LoG;
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

    Mat kernel = calcKernel(5, 3);
    // the values calculated is between 0 and 1 (float), imshow will automatically map them into [0:255]. Since the values are typically small, it'll be shown as almost black
    cout << kernel << endl;
    Mat nkernel;
    normalize(kernel, nkernel, 0, 1, NORM_MINMAX);
    // to show the kernel better, we normalize those value [0,1] into [min,max]
    Mat bigkernel;
    resize(nkernel, bigkernel, Size(500, 500), 0, 0, INTER_NEAREST);
    imshow("minmax_kernel", bigkernel);

    // apply the calculated filter
    Mat out;
    filter2D(img, out, -1, kernel);
    imshow("LoG", out);

    waitKey(0);

    return 0;
}