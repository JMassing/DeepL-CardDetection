#include <iostream>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <sstream>

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define OPENCV

#include "yolo_v2_class.hpp"    // imported functions from Yolov4 DLL

int main(int argc, char* argv[])
{

    // Yolov4 
    std::string names_file = "data/coco.names";
    std::string cfg_file = "cfg/yolov3.cfg";
    std::string weights_file = "yolov3.weights";
    std::string filename;

    if (argc > 3) {    //voc.names yolo-voc.cfg yolo-voc.weights test.mp4
        names_file = argv[1];
        cfg_file = argv[2];
        weights_file = argv[3];
    }
    float const thresh = (argc > 4) ? std::stof(argv[5]) : 0.2;

    std::ifstream names(names_file);
    std::vector<std::string> card_names{};
    if(names.is_open()) {
        std::string name;
        while (std::getline(names, name)) {
            card_names.push_back(name);
        }
        names.close();
    }

    Detector detector(cfg_file, weights_file);

    // OpenCV
    cv::Mat frame;
    cv::VideoCapture capture{};
    int device_id = 0;
    int api_id = cv::CAP_ANY;
    capture.open(device_id, api_id);
    if (!capture.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        capture.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        std::vector<bbox_t> bounding_boxes = detector.detect(frame);

        //draw bounding boxes and probabilities
        cv::Scalar color(0, 255, 0);
        for(auto& bb: bounding_boxes)
        {
            cv::rectangle(frame, cv::Point(bb.x, bb.y), cv::Point(bb.x + bb.w, bb.y + bb.h), color);
            std::stringstream text{};
            text << card_names.at(bb.obj_id) << ": " << std::fixed << std::setprecision(2) << bb.prob*100.0 << "%";
            cv::putText(frame, text.str(), cv::Point(bb.x+bb.w, bb.y), cv::FONT_HERSHEY_PLAIN, 0.75, color, 1.5);
        }        

        // show live and wait for a key with timeout long enough to show images
        cv::imshow("Live", frame);
        if (cv::waitKey(5) >= 0)
            break;
    }
    return 0;
}