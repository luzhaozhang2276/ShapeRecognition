//
// Created by wxq on 2020/10/10.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// #include <opencv2/dnn.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

// TODO: OpenCV的DNN深度神经网络模块，未使用的模块，备用,需要包含头文件: opencv2/dnn.hpp
// using namespace dnn;

const int COLOR_VALUE = 5;
const int FLAG_VALUE = 5;
const double AREA_SIZE = 10.0;

// use 'UNIX like' path, RGBA image
const string INPUT_IMAGE = "data/test.png";
const string OUTPUT_IMAGE = "data/result.png";

//const string INPUT_IMAGE = "1.jpg";
//const string OUTPUT_IMAGE = "output.png";


int main(int argc, char* argv[]) {

    // 读取图片
//    Mat sourceImage = imread(INPUT_IMAGE);
//    Mat inputImage;
//    resize(sourceImage, inputImage, cvSize(640,480));
    Mat inputImage = imread(INPUT_IMAGE);


    // 判断图像是否为空
    if (inputImage.empty()) {
        cout << "源图像读取失败" << endl;
        // 返回错误代码并退出函数
        return -1;
    }

    // 转换图像类型
    Mat __inputImage;
    Mat __tmpImage;
    cvtColor(inputImage, __tmpImage, CV_BGRA2BGR);
    __tmpImage.convertTo(__inputImage, CV_8U);


    // 克隆图像
    // 用于检测圆
    Mat __useCircleImage = __inputImage.clone();
    // 用于检测其他图形
    Mat __useOtherImage = __inputImage.clone();
    // 结果图像
    Mat __resultImage = __inputImage.clone();

    /// 检测圆的算法部分

    // 中值滤波
    Mat __CircleMedianBlurImage;
    medianBlur(__useCircleImage, __CircleMedianBlurImage, 5);

    namedWindow("中值滤波结果", CV_WINDOW_AUTOSIZE);
    imshow("中值滤波结果", __CircleMedianBlurImage);
    waitKey(0);

    // 转换为灰度图像
//    Mat __CircleGrayImage;
//    cvtColor(__CircleMedianBlurImage, __CircleGrayImage, CV_BGR2GRAY);

    // Canny边缘检测
    Mat __CircleCannyImage;
//    Canny(__CircleGrayImage, __CircleCannyImage, 100, 200);
    Canny(__CircleMedianBlurImage, __CircleCannyImage, 100, 200);

    namedWindow("边缘检测结果", CV_WINDOW_AUTOSIZE);
    imshow("边缘检测结果", __CircleCannyImage);
    waitKey(0);

    // 用于存储计算结果
    vector<Vec3f> pcircles;

    // 霍夫圆检测
    HoughCircles(__CircleCannyImage, pcircles, CV_HOUGH_GRADIENT, 1.2, 100, 130, 38, 20, 100);

    // 处理检测结果
    for (size_t i = 0; i < pcircles.size(); i++) {
        // 用于存储当前的圆的识别信息
        Vec3f cc = pcircles[i];

        // 存储BGR三个通道的值
        uchar __targetBlue = __useCircleImage.at<cv::Vec3b>(cc[1], cc[0])[0];
        uchar __targetGreen = __useCircleImage.at<cv::Vec3b>(cc[1], cc[0])[1];
        uchar __targetRed = __useCircleImage.at<cv::Vec3b>(cc[1], cc[0])[2];

        // 绘制文本
        string tempText = "<" + std::to_string((int)(cc[0])) + ", " + std::to_string((int)(cc[1])) + ">";
        putText(__useCircleImage, tempText, Point(cc[0] - 20, cc[1]), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
        putText(__resultImage, tempText, Point(cc[0] - 20, cc[1]), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

        // 绘制圆形和圆的轮廓
        circle(__useCircleImage, Point(cc[0], cc[1]), cc[2], Scalar(255, 0, 0), 2);
        circle(__resultImage, Point(cc[0], cc[1]), cc[2], Scalar(255, 0, 0), 2);
        circle(__useCircleImage, Point(cc[0], cc[1]), 1, Scalar(0, 255, 0), 2);
        circle(__resultImage, Point(cc[0], cc[1]), 1, Scalar(0, 255, 0), 2);
        // 打印提示
        cout << "位于Point<" << std::to_string((int)(cc[0])) << ", " << std::to_string((int)(cc[1])) << ">" << "存在圆形, " << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;
    }
    namedWindow("圆检测结果", CV_WINDOW_AUTOSIZE);
    imshow("圆检测结果", __useCircleImage);
    waitKey(0);

    /// 检测五边形，六边形、三角形

    // 转换为灰度图像
    Mat __OtherGrayImage;
    cvtColor(__useOtherImage, __OtherGrayImage, CV_BGR2GRAY);

    // 图像二值化
    Mat __OtherThresholdImage;
    threshold(__OtherGrayImage, __OtherThresholdImage, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // namedWindow("二值化结果", CV_WINDOW_AUTOSIZE);
    // imshow("二值化结果", __OtherThresholdImage);

    // 用于存储Point的向量
    vector<vector<Point>> __contours;
    vector<Vec4i> __hierarchy;

    // 轮廓发现
    findContours(__OtherThresholdImage, __contours, __hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<vector<Point>> __contourPloy(__contours.size());

    // 定义三种颜色
    const Scalar blue = Scalar(255, 0, 0);
    const Scalar green = Scalar(0, 255, 0);
    const Scalar red = Scalar(0, 0, 255);

    // 用于计算色彩差的阈值
    int __colorValue = COLOR_VALUE;

    // 处理发现的轮廓
    for (size_t i = 0; i < __contours.size(); i++) {
        // 计算轮廓周长
        double __epsilon = 0.02 * arcLength(__contours[i], true);

        // 多边拟合
        approxPolyDP(Mat(__contours[i]), __contourPloy[i], __epsilon, true);

        // 计算中心距
        Moments moment;
        moment = moments(__contours[i]);
        int x = (int)(moment.m10 / moment.m00);
        int y = (int)(moment.m01 / moment.m00);

        // 三角形检测
        if (__contourPloy[i].size() == 3) {
            // 绘制轮廓
            drawContours(__useOtherImage, __contours, i, blue, 2);
            drawContours(__resultImage, __contours, i, blue, 2);

            // 绘制中心点坐标文本
            string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
            putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
            putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

            // 提取中心点颜色
            uchar __targetBlue = __useOtherImage.at<cv::Vec3b>(y, x)[0];
            uchar __targetGreen = __useOtherImage.at<cv::Vec3b>(y, x)[1];
            uchar __targetRed = __useOtherImage.at<cv::Vec3b>(y, x)[2];
            // 绘制中心点
            circle(__useOtherImage, Point(x, y), 2, red, -1);
            circle(__resultImage, Point(x, y), 2, red, -1);

            cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
            cout << "存在三角形, ";
            cout << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;

            // 四边形检测
        } else if (__contourPloy[i].size() == 4) {

            // 四个坐标点的颜色信息
            uchar __bottomBlue = __useOtherImage.at<cv::Vec3b>(y + 20, x)[0];
            uchar __bottomGreen = __useOtherImage.at<cv::Vec3b>(y + 20, x)[1];
            uchar __bottomRed = __useOtherImage.at<cv::Vec3b>(y + 20, x)[2];
            circle(__useOtherImage, Point(x, y + 20), 2, green, -1);
            circle(__resultImage, Point(x, y + 20), 2, green, -1);

            uchar __topBlue = __useOtherImage.at<cv::Vec3b>(y - 20, x)[0];
            uchar __topGreen = __useOtherImage.at<cv::Vec3b>(y - 20, x)[1];
            uchar __topRed = __useOtherImage.at<cv::Vec3b>(y - 20, x)[2];
            circle(__useOtherImage, Point(x, y - 20), 2, green, -1);
            circle(__resultImage, Point(x, y - 20), 2, green, -1);

            uchar __leftBlue = __useOtherImage.at<cv::Vec3b>(y, x + 20)[0];
            uchar __leftGreen = __useOtherImage.at<cv::Vec3b>(y, x + 20)[1];
            uchar __leftRed = __useOtherImage.at<cv::Vec3b>(y, x + 20)[2];
            circle(__useOtherImage, Point(x + 20, y), 2, green, -1);
            circle(__resultImage, Point(x + 20, y), 2, green, -1);

            uchar __rightBlue = __useOtherImage.at<cv::Vec3b>(y + 10, x - 20)[0];
            uchar __rightGreen = __useOtherImage.at<cv::Vec3b>(y + 10, x - 20)[1];
            uchar __rightRed = __useOtherImage.at<cv::Vec3b>(y + 10, x - 20)[2];
            circle(__useOtherImage, Point(x - 20, y), 2, green, -1);
            circle(__resultImage, Point(x - 20, y), 2, green, -1);

            // 计算色差
            int __bottomTopBlue = abs((int)(__bottomBlue)-(int)(__topBlue));
            int __bottomTopGreen = abs((int)(__bottomGreen)-(int)(__topGreen));
            int __bottomTopRed = abs((int)(__bottomRed)-(int)(__topRed));

            int __leftRightBlue = abs((int)(__leftBlue)-(int)(__rightBlue));
            int __leftRightGreen = abs((int)(__leftGreen)-(int)(__rightGreen));
            int __leftRightRed = abs((int)(__leftRed)-(int)(__rightRed));

            // 判断是否存在色差大于__colorValue
            bool __bottomTopColor = (__bottomTopBlue > __colorValue) || (__bottomTopGreen > __colorValue) || (__bottomTopRed > __colorValue);
            bool __leftRightColor = (__leftRightBlue > __colorValue) || (__leftRightGreen > __colorValue) || (__leftRightRed > __colorValue);

            // 存在特殊四边形
            if (__bottomTopColor || __leftRightColor) {
                // 绘制轮廓
                drawContours(__useOtherImage, __contours, i, blue, 2);
                drawContours(__resultImage, __contours, i, blue, 2);

                // 显示坐标点文本
                string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
                putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
                putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

                // 显示信息
                cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
                cout << "存在特殊四边形, 中心点位置: Point(" << std::to_string(x) << ", " << std::to_string(y) << ")" << endl;
                if (__bottomTopColor) {
                    cout << "|__测试点(上)颜色Color(" << std::to_string((int)__topRed) << ", " << std::to_string((int)__topGreen) << ", " << std::to_string((int)__topBlue) << ")" << endl;
                    cout << "|__测试点(下)颜色Color(" << std::to_string((int)__bottomRed) << ", " << std::to_string((int)__bottomGreen) << ", " << std::to_string((int)__bottomBlue) << ")" << endl;
                }
                if (__leftRightColor) {
                    cout << "|__测试点(左)颜色Color(" << std::to_string((int)__leftRed) << ", " << std::to_string((int)__leftGreen) << ", " << std::to_string((int)__leftRed) << ")" << endl;
                    cout << "|__测试点(右)颜色Color(" << std::to_string((int)__rightBlue) << ", " << std::to_string((int)__rightGreen) << ", " << std::to_string((int)__rightRed) << ")" << endl;
                }

                // 判断为矩形
            } else if ((!__bottomTopColor) && (!__leftRightColor)) {
                // 绘制轮廓
                drawContours(__useOtherImage, __contours, i, blue, 2);
                drawContours(__resultImage, __contours, i, blue, 2);
                // 显示信息
                cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
                // 绘制中心点坐标信息
                string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
                putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
                putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

                // 第一个点
                int tempx1 = __contours[i][0].x;
                int tempy1 = __contours[i][0].y;

                // 第一个点与中心点的差距
                int __flagValue = FLAG_VALUE;

                // 计算第一个点的col与中心点的col的差值
                bool flagx1 = abs(tempx1 - x) >= __flagValue ? true : false;

                // 计算第一个点的row与中心点的row的差值
                bool flagy1 = abs(tempy1 - y) >= __flagValue ? true : false;

                // 计算第一个点到中心点的差值
                bool flagxy1 = abs(abs(tempx1 - x) - abs(tempy1 - y)) >= __flagValue ? true : false;

                // 表明中心点与第一个点的X距离之差大于flagValue
                if (flagx1 && (!flagy1)) {
                    cout << "存在菱形, ";
                }

                // 表明中心点与第一个点的XY距离之差大于flagValue
                if (flagx1 && flagy1) {
                    if (flagxy1) {
                        cout << "存在矩形<长方形>, ";
                    } else {
                        cout << "存在矩形<正方形>, ";
                    }

                }

                // 表明中心点与第一个点的Y距离之差大于flagValue
                if ((!flagx1) && flagy1) {
                    cout << "存在菱形, ";
                }
                // 绘制第一个点为位置
                circle(__useOtherImage, Point(tempx1, tempy1), 2, red, -1);

                // 提取颜色， 显示信息
                uchar __targetBlue = __useOtherImage.at<cv::Vec3b>(y, x)[0];
                uchar __targetGreen = __useOtherImage.at<cv::Vec3b>(y, x)[1];
                uchar __targetRed = __useOtherImage.at<cv::Vec3b>(y, x)[2];
                circle(__useOtherImage, Point(x, y), 2, red, -1);
                circle(__resultImage, Point(x, y), 2, red, -1);
                cout << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;
            }
            // 判断为5边形
        } else if (__contourPloy[i].size() == 5) {
            // 绘制轮廓
            drawContours(__useOtherImage, __contours, i, blue, 2);
            drawContours(__resultImage, __contours, i, blue, 2);

            // 读取中心点颜色
            uchar __targetBlue = __useOtherImage.at<cv::Vec3b>(y, x)[0];
            uchar __targetGreen = __useOtherImage.at<cv::Vec3b>(y, x)[1];
            uchar __targetRed = __useOtherImage.at<cv::Vec3b>(y, x)[2];

            // 绘制中心点
            circle(__useOtherImage, Point(x, y), 2, red, -1);
            circle(__resultImage, Point(x, y), 2, red, -1);

            // 绘制中心点坐标信息
            string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
            putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
            putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

            // 显示信息
            cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
            cout << "存在五边形, ";
            cout << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;
            // 判断为六边形
        } else if (__contourPloy[i].size() == 6) {
            // 绘制轮廓
            drawContours(__useOtherImage, __contours, i, blue, 2);
            drawContours(__resultImage, __contours, i, blue, 2);

            // 读取中心点颜色
            uchar __targetBlue = __useOtherImage.at<cv::Vec3b>(y, x)[0];
            uchar __targetGreen = __useOtherImage.at<cv::Vec3b>(y, x)[1];
            uchar __targetRed = __useOtherImage.at<cv::Vec3b>(y, x)[2];

            // 绘制中心点
            circle(__useOtherImage, Point(x, y), 2, red, -1);
            circle(__resultImage, Point(x, y), 2, red, -1);

            // 绘制中心点坐标信息
            string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
            putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
            putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

            // 显示信息
            cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
            cout << "存在六边形, ";
            cout << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;

            // 判断为圆形或者是星星
        } else if (__contourPloy[i].size() >= 10) {
            // 绘制轮廓
            drawContours(__useOtherImage, __contours, i, blue, 2);
            drawContours(__resultImage, __contours, i, blue, 2);
            // 中心点颜色取值
            uchar __targetBlue = __useOtherImage.at<cv::Vec3b>(y, x)[0];
            uchar __targetGreen = __useOtherImage.at<cv::Vec3b>(y, x)[1];
            uchar __targetRed = __useOtherImage.at<cv::Vec3b>(y, x)[2];
            // 绘制中心点
            circle(__useOtherImage, Point(x, y), 2, red, -1);
            circle(__resultImage, Point(x, y), 2, red, -1);
            // 绘制中心点坐标文本
            string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
            putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
            putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);

            // 获取第一个点
            int tempx = __contours[i][0].x;
            int tempy = __contours[i][0].y;
            circle(__useOtherImage, Point(tempx, tempy), 2, red, -1);
            circle(__resultImage, Point(tempx, tempy), 2, red, -1);

            // 计算中心点到第一个点的距离
            double __computeLength = (int)(abs(sqrt(pow((tempx - x), 2) + pow((tempy - y), 2))));

            // 计算面积
            double __computeArea = (double)(3.1415926525 * pow(__computeLength, 2));

            // 获取面积
            double __conArea = contourArea(__contours[i]);

            // 比较面积
            double __areaSize = AREA_SIZE;
            // cout << "计算的面积=" << __computeArea << endl;
            // cout << "获取的面积=" << __conArea << endl;

            bool __isStar = abs(__conArea - __computeArea) > __areaSize ? true : false;

            // 显示提示信息
            cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
            if (__isStar) {
                cout << "存在星星, ";
            } else {
                cout << "存在圆形, ";
            }

            cout << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;

            // 无法判断或者误判断的图形
        } else {

            // 读取中心点颜色
            uchar __targetBlue = __useOtherImage.at<cv::Vec3b>(y, x)[0];
            uchar __targetGreen = __useOtherImage.at<cv::Vec3b>(y, x)[1];
            uchar __targetRed = __useOtherImage.at<cv::Vec3b>(y, x)[2];

            // 绘制中心点和位置坐标
            /*
            circle(__useOtherImage, Point(x, y), 2, blue, -1);
            circle(__resultImage, Point(x, y), 2, blue, -1);
            string tempText = "<" + std::to_string(x) + ", " + std::to_string(y) + ">";
            putText(__useOtherImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
            putText(__resultImage, tempText, Point(x - 20, y), FONT_HERSHEY_PLAIN, 0.7, Scalar(255, 255, 255), 1, 8, false);
            cout << "位于Point<" << std::to_string(x) << ", " << std::to_string(y) << ">";
            cout << "存在其他图形, ";
            cout << "中心点颜色Color(" << std::to_string(__targetRed) << ", " << std::to_string(__targetGreen) << ", " << std::to_string(__targetBlue) << ")" << endl;
            */
        }
    }

    // namedWindow("通用检测", CV_WINDOW_AUTOSIZE);
    // imshow("通用检测", __useOtherImage);

    // 输出图片

    namedWindow("检测结果", CV_WINDOW_AUTOSIZE);
    imshow("检测结果", __resultImage);

    // 保存图片
    imwrite(OUTPUT_IMAGE, __resultImage);

    // waitKey(0);

    waitKey(0);
    return 0;
}