#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;
namespace fs = std::filesystem;

int main() {
    string videoPath = "D:/Saher Hassaballah/Downloads/Telegram Desktop/REC_0005.mp4";
    string modelPath = "D:/path/to/best.onnx";
    string saveDir = "D:/path/to/save/";  // تأكد إن المسار صحيح

    // تأكد إن مجلد الحفظ موجود
    try {
        if (!fs::exists(saveDir)) {
            fs::create_directories(saveDir);
        }
    } catch (const std::exception &e) {
        cerr << "Failed to create save directory: " << e.what() << endl;
        return -1;
    }

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Cannot open video file: " << videoPath << endl;
        return -1;
    }

    dnn::Net net;
    try {
        net = readNetFromONNX(modelPath);
    } catch (const cv::Exception &e) {
        cerr << "Failed to load ONNX model: " << e.what() << endl;
        return -1;
    }

    // اختياري: اختر backend/target لتحسين الاداء
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    namedWindow("Detection", WINDOW_NORMAL);
    resizeWindow("Detection", 900, 520);

    Mat frame;
    int savedCount = 0;

    Size inputSize(1024, 1024);
    const float confThreshold = 0.6f;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        Mat blob;
        blobFromImage(frame, blob, 1.0 / 255.0, inputSize, Scalar(), true, false);

        net.setInput(blob);
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        for (size_t i = 0; i < outputs.size(); ++i) {
            Mat &out = outputs[i];
            // فرضية: كل صف = [x_center, y_center, width, height, confidence, ...]
            for (int r = 0; r < out.rows; ++r) {
                const float* data = out.ptr<float>(r);
                float x_center = data[0];
                float y_center = data[1];
                float width = data[2];
                float height = data[3];
                float confidence = data[4];

                if (confidence < confThreshold) continue;

                // تحويل للنقاط على الإطار الأصلي
                int left = int((x_center - width / 2.0f) * frame.cols);
                int top = int((y_center - height / 2.0f) * frame.rows);
                int right = int((x_center + width / 2.0f) * frame.cols);
                int bottom = int((y_center + height / 2.0f) * frame.rows);

                // clamp للإطار
                left = std::max(0, left);
                top = std::max(0, top);
                right = std::min(frame.cols - 1, right);
                bottom = std::min(frame.rows - 1, bottom);

                int w = right - left;
                int h = bottom - top;
                if (w <= 0 || h <= 0) continue;

                rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
                putText(frame, "Flag", Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

                // احفظ الـ ROI بدل الصورة كاملة
                try {
                    Mat roi = frame(Rect(left, top, w, h)).clone();
                    string filename = saveDir + "flag_" + to_string(savedCount++) + ".jpg";
                    imwrite(filename, roi);
                    cout << "Saved: " << filename << endl;
                } catch (const cv::Exception &e) {
                    cerr << "Error saving ROI: " << e.what() << endl;
                }
            }
        }

        imshow("Detection", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
