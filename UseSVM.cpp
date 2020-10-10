//opencv2.4.13
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
using namespace cv;
using namespace std;


//全局变量
Size sampleSize(160,160);//样本的大小
int train_samples =10;
int classes = 4;
Mat trainData;
Mat trainClasses;

//申明全局函数
Mat readImageSaveContour(Mat src);
void getData();

int main()
{
    Mat src = imread("data/5.png", 0);
//    Mat src = imread("data/test.png", 0);
    getData();

    //定义SVM
    /*CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);*/

    cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::create();
    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::LINEAR);
    svm_->setDegree(0.1);
    svm_->setGamma(0.1);
    svm_->setCoef0(0.1);
    svm_->setC(1);
    svm_->setNu(0.1);
    svm_->setP(0.1);
//    svm_->setTermCriteria(cvTermCriteria(TermCriteria::MAX_ITER, 20000, 0.0001));
    svm_->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 0.0001));

    Ptr<ml::TrainData> tData = ml::TrainData::create(trainData, ml::ROW_SAMPLE, trainClasses);//训练样本的数据类型必须是CV_32F，标签可以是CV_32S或其他。
    svm_->train(tData);
//    svm_->train(trainData, ml::ROW_SAMPLE, trainClasses);

//    cout << "finish" << endl;
//    return 0;

    //得到图像进行预测
    Mat show;
    cvtColor(src,show,8);//8 表示灰度图到彩色图
    Mat imageWhite;
    threshold(src, imageWhite, 100, 255, 8);
    imageWhite = 255 - imageWhite;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(imageWhite, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (int index = contours.size() - 1; index >= 0; index--)
    {
        Rect rectangleTem = boundingRect(contours[index]);
        Mat image;
        image = Mat::zeros(src.size(), CV_8UC1);
        drawContours(image, contours, index, Scalar(255), 2, 8, hierarchy);
        Mat tem = image(rectangleTem);
        Mat imageNewSize;
        resize(tem, imageNewSize, sampleSize, CV_INTER_LINEAR);
        image.release();
        image = imageNewSize.reshape(1, 1);
        image.convertTo(image, CV_32FC1);
        int response = (int)svm_->predict(image);
        if (response == 0)
        {
            cout << "    circle" << endl;
            string str = "circle";
            putText(show, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
        }
        else if (response == 1)
        {
            cout << "    rectangle" << endl;
            string str = "rectangle";
            putText(show, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
        }
        else if (response == 2)
        {
            cout << "    triangle" << endl;
            string str = "triangle";
            putText(show, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
        }
        else if (response == 3)
        {
            cout << "    cross" << endl;
            string str = "cross";
            putText(show, str, Point(rectangleTem.x + rectangleTem.width / 2, rectangleTem.y + rectangleTem.height / 2),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
        }
    }
    imshow("result",show);
    imwrite("result.png", show);

    waitKey(0);
    return 0;
}


void getData()
{
    trainData.create(train_samples*classes, sampleSize.width*sampleSize.height, CV_32FC1);  //CV_32FC1  CV_32SC1
    trainClasses.create(train_samples*classes, 1, CV_32SC1);
    Mat src_image;
    char file[255];
    int i, j;
    for (i = 0; i<classes; i++)
    {
        for (j = 0; j< train_samples; j++)
        {
            sprintf(file, "./data/samples/s%d/%d.png", i, j);
            src_image = imread(file, 0);
            if (src_image.empty())
            {
                printf("Error: Cant load image %s\n", file);
                //exit(-1);
            }
            Mat image = readImageSaveContour(src_image);
            Mat imageNewSize;
            resize(image, imageNewSize, sampleSize, CV_INTER_LINEAR);
            image.release();
            image = imageNewSize.reshape(1, 1);
            image.convertTo(trainData(Range(i*train_samples + j, i*train_samples + j + 1), Range(0, trainData.cols)), CV_32FC1);
            trainClasses.at<int>(i*train_samples + j, 0) = i;
        }
    }
}

Mat readImageSaveContour(Mat src)
{
    Mat imageWhite;
    threshold(src, imageWhite, 100, 255, 8);
    imageWhite = 255 - imageWhite;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(imageWhite, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //最大轮廓
    double maxarea = 0;
    int maxAreaIdx = 0;
    for (int index = contours.size() - 1; index >= 0; index--)
    {
        double tmparea = fabs(contourArea(contours[index]));
        if (tmparea>maxarea)
        {
            maxarea = tmparea;
            maxAreaIdx = index;
        }
    }
    Rect rectangleTem = boundingRect(contours[maxAreaIdx]);
    Mat image;
    image = Mat::zeros(src.size(), CV_8UC1);
    drawContours(image, contours, 0, Scalar(255), 2, 8, hierarchy);
    //Rect newRectangleTem(rectangleTem.x - 1, rectangleTem.y - 1, rectangleTem.width + 2, rectangleTem.height+2);
    Mat tem = image(rectangleTem);
    return tem;
}


