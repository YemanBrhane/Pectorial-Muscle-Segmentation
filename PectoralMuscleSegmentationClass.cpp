#include "PectoralMuscleSegmentationClass.h"

PectoralMuscleSegmentationClass::PectoralMuscleSegmentationClass(QWidget *parent) :
    QMainWindow(parent) // ,ui(new Ui::PectoralMuscleSegmentationClass)
    {

    // initaialization
    TP = 0;
    TN = 0;
    N = 0;
    segAccuracy = 0.0;
    lowThreshold = 2.1;
}

bool PectoralMuscleSegmentationClass::dirExists(const string &dirName_in)
{
    //cout<<"Images directory checking..."<<endl;

    DWORD ftyp = GetFileAttributesA(dirName_in.c_str());
      if (ftyp == INVALID_FILE_ATTRIBUTES)
          return false;  //something is wrong with your path!

      if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
          return true;   // this is a directory!

      return false;    // this is not a directory!
}

vector<Mat> PectoralMuscleSegmentationClass::getImagesInFolder(string folder, string ext, bool force_gray)
{

    // get all files within folder
        std::vector < std::string > files;
        cv::glob(folder, files);

        // open files that contains 'ext'
        std::vector < cv::Mat > images;
        for(auto & f : files)
        {
            if(f.find(ext) == std::string::npos)
                continue;

            images.push_back(cv::imread(f, force_gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED));
        }

        return images;
}

// ROI extractor
Mat PectoralMuscleSegmentationClass::extractROI(const Mat &imageIn, int n)
{
    //cout<<"Extracting ROI..."<<endl;
    Size size(300,400);

    // output of the function
    Mat imout;

    /*
     * the muscle is found within upper left 65 peercent of row of the and n columns
     * n is upper right corner of breast obtained from mask image
    */
    //
    int imrow = (int)(0.65 * imageIn.rows);
    int imcol = n-200;


    imout.create(imrow, imcol,imageIn.type());

    // copy ROI portion
    imageIn(Rect(0,0,imcol, imrow)).copyTo(imout);

    Mat tem;
    tem = imout.clone();
    cv::resize(tem,tem,size);//resize image
    namedWindow("ROI Image",WINDOW_AUTOSIZE);
    imshow("ROI Image",tem);

    return imout;
}

// Breast upper right corner detector
int PectoralMuscleSegmentationClass::breastUpperRightedge(Mat &maskIn)
{
    //cout<<"Extracting breast upper right corner..."<<endl;

    /**
     * in some of the images they have black portion at the top throughy out the whole column
     * I took try to detect the brest edge at row 50
     */

    uchar* pixelval = maskIn.ptr<uchar>(50);
    for (int i =100;i<maskIn.cols;i++){

        if (pixelval[i]==0){
            return i;
        }
    }
}

// Muscle border detector
Mat PectoralMuscleSegmentationClass::muscleBorderDetector(const Mat &imageIn, double sigma)
{
   // cout<<"Muscle border detection..."<<endl;

    // Canny edge detector parameters
    double highThreshold ;
    highThreshold = lowThreshold *3;

    double Min, Max;
    Size size(300,400);

    // Apply gaussian smoothing

    Mat gaussBlurim;
    gaussBlurim.create(imageIn.size(), imageIn.type());
    int kernelRow = (int)(4*sigma + 1);
    Size ksize( kernelRow, kernelRow) ;
    GaussianBlur(imageIn, gaussBlurim,ksize,sigma);


    // Gaussian Filter: user defined kernell;

    //

    Mat Kernel = (Mat_<float>(5, 5) <<
                  2.0, 4.0, 5.0, 4.0, 2.0,
                  4.0, 9.0, 12.0,9.0, 4.0,
                  5.0, 12.0, 15.0, 12.0, 5.0,
                  4.0, 9.0,12.0, 9.0, 4.0,
                  2.0,4.0, 5.0, 4.0, 2.0
                  );

    Kernel = Kernel/159; // normalize

    Mat imfilter2D;

    // apply again smoothing
    filter2D(gaussBlurim,imfilter2D,-1,Kernel);
    //cout<< imfilter2D.size()<<endl;

    Mat temp;
    temp  = imfilter2D.clone();
    cv::resize(temp,temp,size);//resize image
    namedWindow("Gaussian smoothed image",WINDOW_AUTOSIZE);
    imshow("Gaussian smoothed image",temp);

    cv::minMaxLoc(imfilter2D, &Min, &Max);

    // Convert imfilter2D from CV_16U to 8-bit for canny filter
    Mat imCV8;
    if (Min!=Max)
    {
      imfilter2D -= Min;
      imfilter2D.convertTo(imCV8,CV_8U,255/(Max-Min));
    }
    cv::minMaxLoc(imCV8, &Min, &Max);


    // apply canny edge detection
    Mat cannyIm;
    cannyIm.create(imCV8.size(),imCV8.type());
    Canny( imCV8, cannyIm, lowThreshold, highThreshold,3,true);

    Mat xx;
    xx  = cannyIm.clone();
    cv::resize(xx,xx,size);//resize image
    namedWindow("canny image",WINDOW_AUTOSIZE);
    imshow("canny image",xx);
    //cv::minMaxLoc(cannyIm, &Min, &Max);

    return cannyIm;
}

// upper left Triangular filter
Mat PectoralMuscleSegmentationClass::TriangularFilter(const Mat &image)
{
    //cout<<"Applying Triangular Filter..."<<endl;

    int rows, cols,y1,y2,x1,x2;
    Size size(300, 400);
    double slope, d;

    //double Min, Max;

    rows =  image.rows;
    cols = image.cols;

    // off diagonal line end pointa(0,rows-1) and (cols-1,0)
    x1 = 0;y1=rows-1;
    x2 = cols-1;y2=0;

    // diagonal line slope
    slope = ((double)(y1-y2)/(x1-x2));
    std::vector<int> y;

    // find points on the diagonal
    for (int i=0; i<cols; i++)
    {
        d = (int)round(rows-1 + (slope * i));
        y.push_back(d);
    }

    Mat im= image.clone();
    for (int i = 0; i<rows ; i++)
    {
        for (int j = 0 ; j<cols ;j++)
        {
            // if below off diagonal replace by zero
            if (y[j]<=i)
                im.at<uchar>(i,j) = 0;
                }
    }

    Mat imout= Mat::zeros(image.size(), image.type());


    // scann all column of the image hold only the pixel value at
    // highest row

    for (int i = 0; i<cols ; i++)
    {
    for (int j = rows-1 ; j>=0 ;j--)
    {
        int pixVal = (int)im.at<uchar>(j,i);
        if ( pixVal== 255){
            imout.at<uchar>(j,i) = 255;
            break;
                }
     }
    }

    Mat x;
    x  = imout.clone();
    cv::resize(x,x,size);//resize image
    namedWindow("triangular filter output image",WINDOW_AUTOSIZE);
    imshow("triangular filter output image",x);
    return imout;
}

// Muscle region segmentation
Mat PectoralMuscleSegmentationClass::Segmentation(const Mat &image, const Mat& orginalmage)
{
    //cout<<"Muscle Segmentation..."<<endl;

    Size size(300,400);
    Mat I;

    I  = image.clone();
    /**
     * Suppress the detected edges with in the first 200 rows
     */

    I(Rect(0, 0, image.cols, 200)) = Scalar(0);

    /**
      * idxX = row index , idxY = column index
     */
    std::vector<int> idxX;
    std::vector<int> idxY;

    // find row and column index of edge pixels
    find(I, idxX, idxY);

    Mat src_x, src_y,polyCoef;

     // convert vector to Mat
    src_x = Mat (idxX.size(), 1, CV_32F);
    src_y = Mat (idxY.size(), 1, CV_32F);

    for(int i=0; i<src_x.rows; ++i)
    {
        src_x.at<float>(i, 0) = (float)idxX[i];
        src_y.at<float>(i , 0) = (float)idxY[i];
    }

    // apply curve fitting to get pectorial muscle boundary line approximation
    int order =1;
    polyCoef = Mat(order + 1,1, CV_32F);
    cv::polyfit(src_x,src_y,polyCoef,order);

    // Evaluate the line on the whole image column
    Mat polyXValues, polyYValues;
    polyXValues  = Mat(1,image.cols,CV_16U);
    polyYValues = Mat(1,image.cols,CV_16U);

    for(int i = 0 ; i< image.cols ; i++)
    {
        polyXValues.at<ushort>(0,i) = i;
    }

    // evaluate the line
    polyVal(polyCoef, polyYValues);


    std::vector<int> allowedYidx;
    for (int i = 0 ; i<polyXValues.cols; i++)
    {
        /**
            from the physical apperance of the muscle the boundary can be
            approximated y = mx + b But the value of the the boundary equation
            is decreasing as the the image column increases and some times
            when we evaluate on colums far right we might get negative values
            from polyVal( function above;

            to avoid un wanted(negative values) values:
            since the slope of the line is positive maximum(in image coordinate
            maximum row value on the line)value will be at (0,0). so if (polyYValues.at<ushort>(0,i) >
            polyYValues.at<ushort>(0,0)) break because the line have already reached
            image boundary

        */
        if (polyYValues.at<ushort>(0,i) > polyYValues.at<ushort>(0,0))
        break;
        allowedYidx.push_back(polyYValues.at<ushort>(0,i));
    }

    Mat segmentedImage;
    segmentedImage = Mat::zeros(orginalmage.size(),orginalmage.type());
    int dummy;

    /**
      * make muscle region white
    */

    for (int i =0; i< allowedYidx.size(); i++)
    {
        dummy  = allowedYidx[i];
        for (int j = 0 ; j <=dummy ; j++)
            segmentedImage.at<uchar>(j, i) = 255;
    }


    Mat y;
    y = segmentedImage.clone();
    cv::resize(y,y,size);//resize image
    namedWindow("Segmented image",WINDOW_AUTOSIZE);
    imshow("Segmented image",y);

    return segmentedImage;
}

// same as matlab find index values of matrix
void PectoralMuscleSegmentationClass::find(const Mat &image, std::vector<int> &idxX,
                      std::vector<int> &idxY,float value)
{
    assert(image.cols > 0 && image.rows > 0 && image.channels() == 1 && image.depth() == CV_8U);
       const int M = image.rows;
       const int N = image.cols;
       if (value==0.0)
       {

           for (int m = 0; m < M; ++m)
           {
               const uchar* bin_ptr = image.ptr<uchar>(m);
               for (int n = 0; n < N; ++n) {
                   if (bin_ptr[n] > 0)
                   {
                       idxX.push_back(n);
                       idxY.push_back(m);
                   }
               }
           }
        }
       else
       {
           for (int m = 0; m < M; ++m)
           {
               const uchar* bin_ptr = image.ptr<uchar>(m);
               for (int n = 0; n < N; ++n) {
                   if (bin_ptr[n] ==value)
                   {
                       idxX.push_back(n);
                       idxY.push_back(m);
                   }
               }
           }
       }
}


void PectoralMuscleSegmentationClass::polyVal(const Mat &coeff, Mat &yVal)
{
    for (int i = 0 ; i < yVal.cols ; i++)
    {
        // evaluate y = ax + b
        yVal.at<ushort>(0, i) = (int)round( coeff.at<float>(0) +
                         coeff.at<float>(1) * i);
    }
}

void PectoralMuscleSegmentationClass::calculateAccuracy(Mat &segmented_images, Mat &groundtruth_images,
                            Mat &mask_images, Mat &visualResult)
{
    // prepare visual result (3-channel BGR image initialized to black = (0,0,0) )
    visualResult = cv::Mat(segmented_images.size(), CV_8UC3, cv::Scalar(0,0,0));

    for(int y=0; y<segmented_images.rows; y++)
    {
        //cout<<"Here"<<endl;
        uchar* segData = segmented_images.ptr<uchar>(y);
        uchar* gndData = groundtruth_images.ptr<uchar>(y);
        uchar* mskData = mask_images.ptr<uchar>(y);
        uchar* visData = visualResult.ptr<uchar>(y);

        for(int x=0; x<segmented_images.cols; x++)
        {
        if(mskData[x])
        {
        N++;		// found a new sample within the mask

        if(segData[x] && gndData[x])
        {
        TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)

        // mark with blue
        visData[3*x + 1 ] = 255;
        visData[3*x + 1 ] = 0;
        visData[3*x + 2 ] = 0;
        }
        else if(!segData[x] && !gndData[x])
        {
        TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)

        // mark with gray
        visData[3*x + 0 ] = 128;
        visData[3*x + 1 ] = 128;
        visData[3*x + 2 ] = 128;
        }
        else if(segData[x] && !gndData[x])
        {
        // found a false positive

        // mark with yellow
        visData[3*x + 0 ] = 0;
        visData[3*x + 1 ] = 255;
        visData[3*x + 2 ] = 255;
        }
        else
        {
        // found a false positive

        // mark with red
        visData[3*x + 0 ] = 0;
        visData[3*x + 1 ] = 0;
        visData[3*x + 2 ] = 255;
        }
        }
        }

    //visualResult.push_back(visData);
    }

    Mat y;
    cv::Size size(400,300);
    y = visualResult.clone();
    cv::resize(y,y,size);//resize image
    namedWindow("TPTN separated image",WINDOW_AUTOSIZE);
    imshow("TPTN separated image",y);
}

void PectoralMuscleSegmentationClass::performanceEval(string dataset_path)
{
    std::vector <cv::Mat> images; // input images
    std::vector <cv::Mat> truths; // ground truth images
    std::vector <cv::Mat> masks; // mask images
    Mat visual_results;

    bool isExist = dirExists(dataset_path);

    if (isExist)
        cout<<"Path exists..."<<endl;
    else{
        cout<<"Not valid image path for images"<<endl;
        //return -1;
    }

    cout<<"Images loading..."<<endl;

    images = getImagesInFolder(dataset_path + "images", ".png", false);
    truths = getImagesInFolder(dataset_path + "groundtruth", ".png", true);
    masks  = getImagesInFolder(dataset_path + "mask", ".png", true);
    Mat image, maskImage, truthImage;

    //
    int numImages = 20;
    double sigma = 54.0;
    lowThreshold = 2.2;

    // reset performance values
    TP = 0.0;
    TN = 0.0;
    N  = 0.0;

    for (int k =0 ; k<numImages ; k ++)
    {
        //cout<<k<<endl;
        image = images[k].clone();
        maskImage = masks[k].clone();
        truthImage = truths[k].clone();

        int rightCorner = breastUpperRightedge(maskImage);
        Mat imageROI =extractROI(image,rightCorner);

        Mat cannyim;

        cannyim = muscleBorderDetector(imageROI,sigma);

        Mat Im2, ImOut;
        Im2 = TriangularFilter(cannyim);
        ImOut  = Segmentation(Im2, image);

        calculateAccuracy(ImOut,truthImage,  maskImage, visual_results);

        //cout<<i<<"th image accuracy"<<w.segAccuracy<<endl;
        }
        segAccuracy  = (double) (TP + TN) / N;
        cout<<" Over all accuracy is "<< segAccuracy <<endl;
}
