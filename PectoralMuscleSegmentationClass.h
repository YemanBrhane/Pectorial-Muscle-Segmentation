#ifndef PectoralMuscleSegmentationClass_H
#define PectoralMuscleSegmentationClass_H
#include <QMainWindow>

#include <iostream>
#include <QFileDialog>

#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include <windows.h>
#include <Shlwapi.h>

#include <math.h>
#include <vector>

//#include <filesystem> // C++17 (or Microsoft-specific implementation in C++14)

using namespace std;
using namespace cv;

class PectoralMuscleSegmentationClass : public QMainWindow
{
    Q_OBJECT

public:
    explicit PectoralMuscleSegmentationClass(QWidget *parent = 0);

    int TP; //Number of True positive
    int TN; //Number of True negative
    int N;  //Number of pixels under consideration

    double segAccuracy; // segegmentation accuracy

    double lowThreshold; // canny edge detection lower treshold value

    /**
        Returns True or false depending on wheater the direciroy exists or not.

        @param dirName_in  Directory name to be checked.
        @return True of directory exist, False if directory does not exist.
    */
    bool dirExists(const std::string& dirName_in);

    /**
        Returns vector of images which are contained all the images in a folder.

        @param folder folder or directory.
        @param ext filtering parameter, image extension(.png, .gif, ...).
        @param force_gray boolean to read the image as gray scale image or
        in its orginal format.
        @return vector of images of all the images in the folder specified
    */
    vector <cv::Mat> getImagesInFolder(string folder,
                                       string ext, bool force_gray);

    /**
        Returns right upper corner of input image from mask image given

        @param maskIn input image.
        @return column number of right upper most of image
    */
    int breastUpperRightedge(Mat& maskIn);

    /**
        Returns region which contains pectorial muscle.

        @param imageIn input image.
        @param n right most corner of the of the region to be extracted.
        @return region of interest image.
    */
    Mat extractROI(const Mat& imageIn,int n);

    /**
        Returns edge image with pectorial muscle border detected

        @param imageIn The radius of the circle.
        @param Sigma standard deviation value for gaussian smoothing.
        @return edge image.
    */
    Mat muscleBorderDetector (const Mat& imageIn,double Sigma);

    /**
        Returns an image with all pixels right of forward diagonal zero.

        @param  image region of interest image.
        @return image all pixels right of forward diagonal suppressed.
    */
    Mat TriangularFilter(const Mat& image);


    /**
        Returns returns segmented image.

        @param roiImage region of interest image.
        @param imageIn orginal image to put the segmentation.
        @return  segmented image.
    */
    Mat Segmentation(const Mat& roiImage, const Mat& imageIn);

    /**
        Returns row and column index of pixels greater depending on the    VALUE input.

        @param image input image to be scanned for inde.
        @param idx vector to hold row index.
        @param idxY vector to hold column index
        @param value pixel value you want to know its index, default 0.0(all pixels >0)
        @return void
    */
    void find(const cv::Mat& image, std::vector<int> &idxX,
              std::vector<int> &idxY, float value=0.0);


    /**
        It evaluates a polynomial y = ax  + b

        @param coeff coeeficientn(b , a) of line y = ax +b from polyfit.
        @param values on which hold evaluated values(its size should be specified during call).
        @return void.
    */
    void polyVal(const cv::Mat& coeff, cv::Mat& values);

    /**
        Calculates TP, TN, FP, FN and accuracy of segmentation;

        @param segmented_images segmented image.
        @param groundtruth_images ground tructh image
        @param mask_images mask image
        @param visual_results colored image of true positive=blue,
        true negative:= gray, false positive=yelow, false negative: red
        @return void
    */
    void calculateAccuracy( Mat & segmented_images, Mat & groundtruth_images,
       Mat & mask_images, Mat & visual_results	);


    /**
        It evaluates overall accuracy on the whole data found in the directory
        given as an input

        @param dataset_path directory where images found.
        @return void
    */
    void performanceEval(string dataset_path);

};

#endif // PectoralMuscleSegmentationClass_H
