#include "PectoralMuscleSegmentationClass.h"
#include <QApplication>

int main(int argc, char *argv[]) {

    QApplication app(argc, argv);
    PectoralMuscleSegmentationClass w;

    Mat visual_results; // mask images
    Mat image;
    Mat mask;
    Mat truth;

    string dataset_path = "../dataset/";

   int test = 1;
   /**
     * if test=1, it will test on a single image specified below
     * if test!=1, on the whole dataset found on the dataset_path directory above
    */

    if (test != 1)

        w.performanceEval(dataset_path);
    else{

        bool isExist = w.dirExists(dataset_path);

        if (isExist)
        cout<<"Path exists..."<<endl;
        else{
        cout<<"Not valid image path for images"<<endl;
        return -1;
        }

        image = imread("../dataset/images/15.png",CV_LOAD_IMAGE_UNCHANGED);

        truth = imread("../dataset/groundtruth/15.png", CV_LOAD_IMAGE_GRAYSCALE);

        mask = imread("../dataset/mask/15.png", CV_LOAD_IMAGE_GRAYSCALE);

        double sigma = 54.0;
        w.lowThreshold = 2.2;

        w.TP = 0.0;
        w.TN = 0.0;
        w.N  = 0.0;

        int rightCorner = w.breastUpperRightedge(mask);
        Mat imageROI = w.extractROI(image,rightCorner);

        Mat cannyim;

        cannyim = w.muscleBorderDetector(imageROI,sigma);

        Mat Im2, ImOut;
        Im2 = w.TriangularFilter(cannyim);
        ImOut  = w.Segmentation(Im2, image);

        w.calculateAccuracy(ImOut,truth,  mask, visual_results);

        cout<<"Accuracy "<< (double) (w.TP + w.TN) / w.N;
        waitKey(0);
    }
}
