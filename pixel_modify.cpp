#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main()
{
    // create a matrix
    Mat mat = Mat::zeros(Size(5, 5), CV_32F);
    cout << mat <<endl;
    // pointer to scan
    float *ptr;
    int nRows = mat.rows;
    int nCols = mat.cols;
    // by default matrices are created 1d array
    if (mat.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    for (int i = 0; i < nRows; i++)
    {
        // ptr points to the 1st element of each row
        ptr = mat.ptr<float>(i);
        for (int j = 0; j < nCols; j++)
        {
            // modify pixel content
            *ptr = 1;
            // move ptr
            ptr++;
            // alternatively we can keep ptr always pointing to the 1st element of a row and instead assign the jth element e.i. *(ptr+j)=1
        }
    }
    
    cout << mat <<endl;

    return 0;
}