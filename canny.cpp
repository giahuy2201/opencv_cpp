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

MatIterator_<float> nearest(MatIterator_<float> p1, MatIterator_<float> p2, MatIterator_<float> p3, float angle)
{
    float x = cos(angle);
    float y = sin(angle);
    float x1, y1, x2, y2, x3, y3;
    if (angle >= 0 && angle < M_PI / 2)
    {
        x1 = 1;
        y1 = 0;
        x2 = 0;
        y2 = 1;
        x3 = 1;
        y3 = 1;
    }
    else if (angle >= M_PI / 2 && angle < M_PI)
    {
        x1 = -1;
        y1 = 0;
        x2 = 0;
        y2 = 1;
        x3 = -1;
        y3 = 1;
    }
    else if (angle >= M_PI && angle < 3 * M_PI / 2)
    {
        x1 = -1;
        y1 = 0;
        x2 = 0;
        y2 = -1;
        x3 = -1;
        y3 = -1;
    }
    else
    {
        x1 = 1;
        y1 = 0;
        x2 = 0;
        y2 = -1;
        x3 = 1;
        y3 = -1;
    }
    float d1 = sqrt(pow(x - x1, 2) + pow(y - y1, 2));
    float d2 = sqrt(pow(x - x2, 2) + pow(y - y2, 2));
    float d3 = sqrt(pow(x - x3, 2) + pow(y - y3, 2));
    float min = std::min(std::min(d1, d2), d3);
    if (min == d1)
    {
        return p1;
    }
    else if (min == d2)
    {
        return p2;
    }
    else
    {
        return p3;
    }
}

Mat thinning(Mat orient, Mat magnit)
{
    const int nRows = orient.rows;
    const int nCols = orient.cols;
    Mat mat = Mat::zeros(nRows, nCols, CV_32F);
    // Mat mat = magnit.clone();
    // orient += M_PI;
    // orient /= 2 * M_PI;

    MatIterator_<float> itr = mat.begin<float>() + nCols + 1;
    MatIterator_<float> end = mat.end<float>() - nCols - 1;
    MatIterator_<float> itrO = orient.begin<float>() + nCols + 1;
    MatIterator_<float> endO = orient.end<float>() - nCols - 1;
    MatIterator_<float> itrM = magnit.begin<float>() + nCols + 1;
    MatIterator_<float> endM = magnit.end<float>() - nCols - 1;
    for (; itr != end; itr++)
    {
        MatIterator_<float> topleft = itrM - nCols - 1;
        MatIterator_<float> top = itrM - nCols;
        MatIterator_<float> topright = itrM - nCols + 1;
        MatIterator_<float> midleft = itrM - 1;
        MatIterator_<float> mid = itrM;
        MatIterator_<float> midright = itrM + 1;
        MatIterator_<float> botleft = itrM + nCols - 1;
        MatIterator_<float> bot = itrM + nCols;
        MatIterator_<float> botright = itrM + nCols + 1;
        float x = cos(*itrO);
        float y = sin(*itrO);

        if (*itrO >= 0 && *itrO < M_PI / 2)
        {
            float x1 = 0, x2 = 1, y1 = 0, y2 = 1;
            float interpol1 = (*mid * (x2 - x) * (y2 - y) + *midright * (x - x1) * (y2 - y) + *top * (x2 - x) * (y - y1) + *topright * (x - x1) * (y - y1)) / (x2 - x1) / (y2 - y1);
            x1 = -1;
            x2 = 0;
            y1 = -1;
            y2 = 0;
            float interpol2 = (*botleft * (x2 - x) * (y2 - y) + *bot * (x - x1) * (y2 - y) + *midleft * (x2 - x) * (y - y1) + *mid * (x - x1) * (y - y1)) / (x2 - x1) / (y2 - y1);
            if (*mid > interpol1 && *mid > interpol2)
            {
                *itr = *mid;
            }
            // if (*mid > interpol1 && *mid > interpol2)
            // {
            //     *itr = 0;
            // }
            // else
            // {
            //     if (*mid > interpol1)
            //     {
            //         *nearest(midright, top, topright, *itrO) = 0;
            //     }
            //     if (*mid > interpol2)
            //     {
            //         *nearest(midleft, bot, botleft, *itrO) = 0;
            //     }
            // }
        }
        else
        {
            float x1 = -1, x2 = 0, y1 = 0, y2 = 1;
            float interpol1 = (*midleft * (x2 - x) * (y2 - y) + *mid * (x - x1) * (y2 - y) + *topleft * (x2 - x) * (y - y1) + *top * (x - x1) * (y - y1)) / (x2 - x1) / (y2 - y1);
            x1 = 0;
            x2 = 1;
            y1 = -1;
            y2 = 0;
            float interpol2 = (*bot * (x2 - x) * (y2 - y) + *botright * (x - x1) * (y2 - y) + *mid * (x2 - x) * (y - y1) + *midright * (x - x1) * (y - y1)) / (x2 - x1) / (y2 - y1);
            if (*mid > interpol1 && *mid > interpol2)
            {
                *itr = *mid;
            }
            // if (*mid > interpol1 && *mid > interpol2)
            // {
            //     *itr = 0;
            // }
            // else
            // {
            //     if (*mid > interpol1)
            //     {
            //         *nearest(midleft, top, topleft, *itrO) = 0;
            //     }
            //     if (*mid > interpol2)
            //     {
            //         *nearest(midright, bot, botright, *itrO) = 0;
            //     }
            // }
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

    MatIterator_<float> itr = mat.begin<float>() + nCols + 1;
    MatIterator_<float> end = mat.end<float>() - nCols - 1;
    MatIterator_<float> itr2 = src.begin<float>() + nCols + 1;
    MatIterator_<float> end2 = src.end<float>() - nCols - 1;
    for (; itr != end; itr++)
    {
        if (*itr2 >= h)
        {
            *itr = 1;
        }
        else if (*itr2 >= l)
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
                *itr = 1;
            }
        }
        itr2++;
    }
    return mat;
}

int low = 20, high = 30;
Mat mat;

static void onTrackbar(int, void *)
{
    Mat mat2 = hysteresis(mat, low, high);
    imshow("hysteresis", mat2);
}

int main()
{
    Mat img;
    Mat img0 = imread("pikrepo.com.jpg", IMREAD_GRAYSCALE);
    // always convert image to float preserve negative values
    img0.convertTo(img, CV_32F, 1.0 / 255.0);

    imshow("original", img);
    cout << "depth: " << img.depth() << ", channels: " << img.channels() << endl;

    GaussianBlur(img, img, Size(5, 5), 2);

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

    string wdn = "hysteresis";
    namedWindow(wdn);
    createTrackbar("low", wdn, &low, 255, onTrackbar);
    createTrackbar("high", wdn, &high, 255, onTrackbar);
    mat = thinn;

    // Mat out;
    // Canny(img0, out, 50, 150);
    // imshow("canny", out);

    onTrackbar(0, 0);

    waitKey(0);

    return 0;
}