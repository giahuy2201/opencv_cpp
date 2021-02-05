#include <iostream>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

void showKernel(Mat kernel)
{
    Mat bigK;
    resize(kernel, bigK, Size(kernel.rows * 100, kernel.cols * 100), 0, 0, INTER_NEAREST);
    imshow("kernel", bigK);
}

void showOrient(Mat orient)
{
    const int nRows = orient.rows;
    const int nCols = orient.cols;
    Mat mat(nRows, nCols, CV_32FC3);

    MatIterator_<Vec3f> itr = mat.begin<Vec3f>();
    MatIterator_<Vec3f> end = mat.end<Vec3f>();
    MatIterator_<float> itr2 = orient.begin<float>();
    MatIterator_<float> end2 = orient.end<float>();
    for (; itr != end && itr2 != end2; itr++)
    {
        float angle = *itr2++;
        (*itr)[1] = sin(angle);
        (*itr)[2] = cos(angle);
    }

    normalize(orient, orient, 0.0, 1.0, NORM_MINMAX, -1);
    imshow("orientation", mat);
}

Mat sobelXk()
{
    Mat mat = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    return mat;
}

Mat sobelYk()
{
    Mat mat = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    return mat;
}

Mat orientation(Mat sobelx, Mat sobely)
{
    const int nRows = sobelx.rows;
    const int nCols = sobelx.cols;
    Mat mat(nRows, nCols, CV_32F);

    for (int i = 0; i < nRows; i++)
    {
        MatIterator_<float> itr = mat.row(i).begin<float>();
        MatIterator_<float> end = mat.row(i).end<float>();
        for (int j = 0; j < nCols && itr != end; j++)
        {
            float gx = sobelx.at<float>(i, j);
            float gy = sobely.at<float>(i, j);
            *itr++ = atan2(gy, gx);
        }
    }
    return mat;
}

Mat magnitude(Mat sobelx, Mat sobely)
{
    const int nRows = sobelx.rows;
    const int nCols = sobelx.cols;
    Mat mat(nRows, nCols, CV_32F);

    for (int i = 0; i < nRows; i++)
    {
        MatIterator_<float> itr = mat.row(i).begin<float>();
        MatIterator_<float> end = mat.row(i).end<float>();
        for (int j = 0; j < nCols && itr != end; j++)
        {
            float gx = sobelx.at<float>(i, j);
            float gy = sobely.at<float>(i, j);
            *itr++ = sqrt(pow(gx, 2) + pow(gy, 2));
        }
    }
    return mat;
}

Mat binary(Mat src)
{
    const int nRows = src.rows;
    const int nCols = src.cols;
    Mat mat(nRows, nCols, CV_32F);

    MatIterator_<float> itr = mat.begin<float>();
    MatIterator_<float> end = mat.end<float>();
    MatIterator_<float> itr2 = src.begin<float>();
    MatIterator_<float> end2 = src.end<float>();
    for (; itr != end && itr2 != end2; itr++)
    {
        if (*itr2++ > 0)
        {
            *itr = 1;
        }
        else
        {
            *itr = 0;
        }
    }
    return mat;
}

Mat thinning(Mat orient, Mat magnit)
{
    const int nRows = orient.rows;
    const int nCols = orient.cols;
    Mat mat = Mat::zeros(nRows, nCols, CV_32F);

    normalize(orient, orient, 0, 1, NORM_MINMAX);
    orient *= 360;
    float sl = 360 / 16.;

    MatIterator_<float> itr = mat.begin<float>() + 1;
    MatIterator_<float> end = mat.end<float>() - 1;
    MatIterator_<float> itrO = orient.begin<float>() + 1;
    MatIterator_<float> endO = orient.end<float>() - 1;
    MatIterator_<float> itrM = magnit.begin<float>() + 1;
    MatIterator_<float> endM = magnit.end<float>() - 1;
    for (; itr != end; itr++)
    {
        float ngbr1, ngbr2;
        if (*itrO >= 3 * sl && *itrO < 5 * sl)
        {
            ngbr1 = *(itrM - nCols + 1);
            ngbr2 = *(itrM + nCols - 1);
        }
        else if (*itrO >= 5 * sl && *itrO < 7 * sl)
        {
            ngbr1 = *(itrM - nCols);
            ngbr2 = *(itrM + nCols);
        }
        else if (*itrO >= 7 * sl && *itrO < 9 * sl)
        {
            ngbr1 = *(itrM - nCols - 1);
            ngbr2 = *(itrM + nCols + 1);
        }
        else
        {
            ngbr1 = *(itrM - 1);
            ngbr2 = *(itrM + 1);
        }

        if (*itrM > ngbr1 && *itrM > ngbr2)
        {
            *itr = *itrM;
        }
        itrO++;
        itrM++;
    }
    return mat;
}

Mat hysteresis(Mat src, int low, int high)
{
    float l = low / 255.;
    float h = high / 255.;
    const int nRows = src.rows;
    const int nCols = src.cols;
    Mat mat = Mat::zeros(nRows, nCols, CV_32F);

    MatIterator_<float> itr = mat.begin<float>() + 1;
    MatIterator_<float> end = mat.end<float>() - 1;
    MatIterator_<float> itr2 = src.begin<float>() + 1;
    MatIterator_<float> end2 = src.end<float>() - 1;
    for (; itr != end && itr2 != end2; itr++)
    {
        if (*itr2 >= h)
        {
            *itr = *itr2;
        }
        else
        {
            float ngbr1 = *(itr2 - nCols - 1);
            float ngbr2 = *(itr2 - nCols);
            float ngbr3 = *(itr2 - nCols + 1);
            float ngbr4 = *(itr2 - 1);
            float ngbr6 = *(itr2 + 1);
            float ngbr7 = *(itr2 + nCols - 1);
            float ngbr8 = *(itr2 + nCols);
            float ngbr9 = *(itr2 + nCols + 1);
            if (ngbr1 >= h || ngbr2 >= h || ngbr3 >= h || ngbr4 >= h || ngbr6 >= h || ngbr7 >= h || ngbr8 >= h || ngbr9 >= h)
            {
                *itr = *itr2;
            }
        }
        itr2++;
    }
    return mat;
}

int main()
{
    Mat img;
    Mat img0 = imread("pikrepo.com.jpg", IMREAD_GRAYSCALE);
    // always convert image to float preserve negative values
    img0.convertTo(img, CV_32F, 1.0 / 255.0);

    imshow("original", img);
    cout << "depth: " << img.depth() << ", channels: " << img.channels() << endl;

    GaussianBlur(img, img, Size(5, 5), 3);

    Mat sbx = sobelXk();
    Mat sobelx, nsobelx;
    filter2D(img, sobelx, -1, sbx);
    // there are negative values when applying sobel, so
    normalize(sobelx, nsobelx, 0.0, 1.0, NORM_MINMAX, -1);
    imshow("sobel x", nsobelx);

    Mat sby = sobelYk();
    Mat sobely, nsobely;
    filter2D(img, sobely, -1, sby);
    normalize(sobely, nsobely, 0.0, 1.0, NORM_MINMAX, -1);
    imshow("sobel y", nsobely);

    Mat orient = orientation(sobelx, sobely);
    // normalize(orient, orient, 0.0, 1.0, NORM_MINMAX, -1);
    showOrient(orient);
    // imshow("orientation", orient);

    Mat magnit = magnitude(sobelx, sobely);
    imshow("magnitude", magnit);

    Mat thinn = thinning(orient, magnit);
    imshow("thin", thinn);

    Mat hyst = hysteresis(thinn, 10, 50);
    imshow("hysteresis", hyst);

    Mat bin = binary(hyst);
    imshow("binary", bin);

    Mat out;
    Canny(img0, out, 50, 150);
    imshow("canny", out);

    waitKey(0);

    return 0;
}