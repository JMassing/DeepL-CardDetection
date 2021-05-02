#include <iostream>
#include <stdio.h>

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char* argv[])
{
    cv::Mat frame;
    cv::VideoCapture capture{};
    int device_id = 0;
    int api_id = cv::CAP_ANY;
    capture.open(device_id, api_id);
    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        capture.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);
        if (waitKey(5) >= 0)
            break;
    }
    return 0;
}