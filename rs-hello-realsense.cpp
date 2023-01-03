// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>             // for cout
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <string> 
#include <fstream>
#include <ctime>
#include <windows.h>
// Hello RealSense example demonstrates the basics of connecting to a RealSense device
// and taking advantage of depth data
enum class direction
{
    to_depth,
    to_color
};
using namespace std;
using namespace cv;
using namespace dnn;


int main(int argc, char* argv[]) try
{

    string duration_c = argv[1];
    int duration_i = stoi(duration_c);
    cout << duration_i << endl;

    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;
    std::vector<std::string> classes;
    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;
    rs2::config cfg;
    float dddpoint[3];
    cfg.enable_stream(RS2_STREAM_DEPTH);
    cfg.enable_stream(RS2_STREAM_COLOR);

    std::ifstream file("coco.names");
    std::string line;
    DetectionModel model;
    std::vector<int> classIds;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;

    Point pt;
    rs2::points points;
    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    int x_pix, y_pix;
    /*cout << points.get_data_size() << endl;*/
    while (std::getline(file, line))
    {
        classes.push_back(line);
    }

    Net net = readNetFromDarknet("C:\\workspace\\ConerRadar\\yolov4_tiny.cfg", "C:\\workspace\\ConerRadar\\yolov4_tiny.weights");
    /*  net.setPreferableBackend(DNN_BACKEND_CUDA);
      net.setPreferableTarget(DNN_TARGET_CUDA);*/
    model = DetectionModel(net);

    model.setInputParams(1 / 255.0, Size(608, 608), Scalar(), true);
    // Configure and start the pipeline
    p.start(cfg);
    //cv::Mat cv_mat;
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    // ...B
    direction   dir = direction::to_color;  // Alignment direction
    using namespace cv;
    const auto window_name_1 = "Display Depth Image";
    const auto window_name_2 = "Display Color Image";
    namedWindow(window_name_1, WINDOW_AUTOSIZE);
    namedWindow(window_name_2, WINDOW_AUTOSIZE);
    SYSTEMTIME sys;
    int duration;
    char start[64] = { NULL };
    char now[64] = { NULL };
    GetLocalTime(&sys);
    sprintf(start, "%4d-%02d-%02d %02d:%02d:%02d ms:%03d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds);
    ofstream OutFile("Target_position.txt"); //
    int begin = sys.wMinute * 1000 * 60 + sys.wSecond * 1000 + sys.wMilliseconds;

    while (waitKey(1) < 0 && getWindowProperty(window_name_1, WND_PROP_AUTOSIZE) >= 0 && getWindowProperty(window_name_2, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = p.wait_for_frames(); // Wait for next set of frames from the camera
        GetLocalTime(&sys);
        cout << &sys << endl;
        if (dir == direction::to_depth)
        {
            // Align all frames to depth viewport
            data = align_to_depth.process(data);
        }
        else
        {
            // Align all frames to color viewport
            data = align_to_color.process(data);
        }
        rs2::depth_frame depth_1 = data.get_depth_frame();
        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame color = data.get_color_frame();
        // Query frame size (width and height)
        const int w_d = depth.as<rs2::video_frame>().get_width();
        const int h_d = depth.as<rs2::video_frame>().get_height();
        const int w_c = color.as<rs2::video_frame>().get_width();
        const int h_c = color.as<rs2::video_frame>().get_height();
        // Create OpenCV matrix of size (w,h) from the colorized depth data
        auto inrist = rs2::video_stream_profile(depth.get_profile()).get_intrinsics();

        // Generate the pointcloud and texture mappings
       /* points = pc.calculate(depth_1);*/
        /*std::cout << points.get_texture_coordinates() << std::endl;*/

        Mat image_d(Size(w_d, h_d), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
        Mat image_c(Size(w_c, h_c), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
        model.detect(image_c, classIds, scores, boxes, 0.6, 0.4);
        //std::cout << "cv"<<image_c.size << std::endl;//height * widthbb
        //std::cout << "color_w" << w_c << std::endl;
        //std::cout << "color_h" << h_c << std::endl;
        float distance = 0;

        for (int i = 0; i < classIds.size(); i++)
        {
            if (classIds[i] == 0) {
                rectangle(image_c, boxes[i], Scalar(0, 255, 0), 2);

                pt.x = boxes[i].x - 10;
                pt.y = boxes[i].y - 10;
                x_pix = boxes[i].x + boxes[i].width / 2;
                y_pix = boxes[i].y + boxes[i].height / 2;
                distance = depth_1.get_distance(x_pix, y_pix);
                float pix[2] = { x_pix,y_pix };
                rs2_deproject_pixel_to_point(dddpoint, &inrist, pix, distance);
                //putText(image_c, std::to_string(distance)+"m", pt, FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
                putText(image_c, "position:" + std::to_string(dddpoint[0]) + "," + std::to_string(dddpoint[1]) + "," + std::to_string(dddpoint[2]), pt, FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
                // std::cout << "position:" + std::to_string(dddpoint[0]) + "," + std::to_string(dddpoint[1]) + "," + std::to_string(dddpoint[2]) << std::endl;
                sprintf(now, "%4d-%02d-%02d %02d:%02d:%02d ms:%03d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds);
                OutFile << now << "position:" + std::to_string(dddpoint[0]) + "," + std::to_string(dddpoint[1]) + "," + std::to_string(dddpoint[2]) << endl; //writing information into Target_position.txt file.


            }

        }


        // Update the window with new data
        imshow(window_name_1, image_d);
        imshow(window_name_2, image_c);
        duration = sys.wMinute * 1000 * 60 + sys.wSecond * 1000 + sys.wMilliseconds;
        if (duration > 1000 * duration_i + begin)
        {
            OutFile.close(); //¹Ø±ÕTest.txtÎÄ¼þ
            break;
        }
    }

    return EXIT_SUCCESS;

    //while (true)
    //{   
    // 
    //   
    //    // Block program until frames arrive
    //    rs2::frameset frames = p.wait_for_frames();

    //    // Try to get a frame of a depth image
    //    rs2::depth_frame depth = frames.get_depth_frame();

    //    // Get the depth frame's dimensions
    //    auto width = depth.get_width();
    //    auto height = depth.get_height();

    //    // Query the distance from the camera to the object in the center of the image
    //    float dist_to_center = depth.get_distance(width / 2, height / 2);

    //    // Print the distance
    //    std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";
    //}

    //return EXIT_SUCCESS;
}



catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
