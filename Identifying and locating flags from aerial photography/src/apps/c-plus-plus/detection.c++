// main.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/telemetry/telemetry.h>

#include <GeographicLib/LocalCartesian.hpp>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <atomic>

// ---------- CONFIG ----------
const std::string VIDEO_PATH = "rtsp://192.168.144.25:8554/main.264";
const std::string MODEL_ONNX = "best.onnx"; // convert your .pt -> .onnx first

// Camera intrinsics (example - replace with your calibrated values)
cv::Mat K = (cv::Mat_<double>(3, 3) << 10600.0, 0.0, 1920.0,
             0.0, 10600.0, 1080.0,
             0.0, 0.0, 1.0);
cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

// Camera offset relative to drone body (meters) (x = forward/east, y = right/north, z = up)
cv::Vec3d cam_trans_offset = {0.0, 0.0, 0.0};

// Rotation offset from camera frame to body frame (3x3) - default identity
cv::Mat cam_rot_offset = cv::Mat::eye(3, 3, CV_64F);

// ground altitude (meters MSL or chosen ref)
double ground_alt = 0.0;
// ----------------------------

// thread-safe frame queue
std::deque<cv::Mat> frameQueue;
std::mutex qMutex;
std::condition_variable qCond;
const size_t MAX_QUEUE = 5;
std::atomic<bool> running(true);

// telemetry shared
struct Telemetry
{
    double lat = 0.0, lon = 0.0, alt = 0.0;    // alt in meters (MSL or AGL depending on MAV)
    double roll = 0.0, pitch = 0.0, yaw = 0.0; // radians
    std::mutex m;
} telemetry;

// helper: rotation matrix from roll,pitch,yaw (body->ENU)
cv::Mat rotationMatrixFromEuler(double roll, double pitch, double yaw)
{
    double cr = cos(roll), sr = sin(roll);
    double cp = cos(pitch), sp = sin(pitch);
    double cy = cos(yaw), sy = sin(yaw);

    cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                   0, cr, -sr,
                   0, sr, cr);

    cv::Mat R_y = (cv::Mat_<double>(3, 3) << cp, 0, sp,
                   0, 1, 0,
                   -sp, 0, cp);

    cv::Mat R_z = (cv::Mat_<double>(3, 3) << cy, -sy, 0,
                   sy, cy, 0,
                   0, 0, 1);

    return R_z * R_y * R_x; // yaw -> pitch -> roll
}

// pixel -> geo (lat,lon) using telemetry and camera geometry
bool pixelToGeo(int cx, int cy, double &out_lat, double &out_lon)
{
    // undistort points to normalized coordinates
    std::vector<cv::Point2f> pts_in = {cv::Point2f((float)cx, (float)cy)};
    std::vector<cv::Point2f> pts_out;
    cv::undistortPoints(pts_in, pts_out, K, distCoeffs); // returns normalized (x/z, y/z)
    double x = pts_out[0].x, y = pts_out[0].y;
    cv::Mat ray_cam = (cv::Mat_<double>(3, 1) << x, y, 1.0);

    // apply camera->body rotation
    cv::Mat ray_body = cam_rot_offset * ray_cam;

    // read telemetry snapshot
    double lat, lon, alt, roll, pitch, yaw;
    {
        std::lock_guard<std::mutex> lk(telemetry.m);
        lat = telemetry.lat;
        lon = telemetry.lon;
        alt = telemetry.alt;
        roll = telemetry.roll;
        pitch = telemetry.pitch;
        yaw = telemetry.yaw;
    }

    // rotate to ENU/world
    cv::Mat R_body_to_enu = rotationMatrixFromEuler(roll, pitch, yaw);
    cv::Mat ray_enu = R_body_to_enu * ray_body; // [east; north; up]

    // camera altitude
    double cam_alt = alt + cam_trans_offset[2];
    double ez = ray_enu.at<double>(2, 0);
    if (fabs(ez) < 1e-8)
        return false; // parallel

    double scale = (cam_alt - ground_alt) / ez;
    double east = ray_enu.at<double>(0, 0) * scale;
    double north = ray_enu.at<double>(1, 0) * scale;

    // convert east/north meters to lat/lon using GeographicLib LocalCartesian
    GeographicLib::LocalCartesian proj(lat, lon, cam_alt); // local reference
    double lat2, lon2, h2;
    proj.Reverse(east, north, 0.0, lat2, lon2, h2); // inputs are X=E, Y=N

    out_lat = lat2;
    out_lon = lon2;
    return true;
}

// video reader thread
void readerThread(const std::string &source)
{
    cv::VideoCapture cap(source);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open stream: " << source << "\n";
        running = false;
        return;
    }
    while (running)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        {
            std::unique_lock<std::mutex> lk(qMutex);
            if (frameQueue.size() >= MAX_QUEUE)
            {
                frameQueue.pop_front();
            }
            frameQueue.push_back(frame);
        }
        qCond.notify_one();
    }
    cap.release();
}

// telemetry listener using MAVSDK
void mavsdkThread(const std::string &connection_url)
{
    mavsdk::Mavsdk mavsdk;
    mavsdk::ConnectionResult conres = mavsdk.add_any_connection(connection_url);
    if (conres != mavsdk::ConnectionResult::Success)
    {
        std::cerr << "MAVSDK connection failed: " << (int)conres << "\n";
        running = false;
        return;
    }

    // wait for system
    std::shared_ptr<mavsdk::System> system;
    {
        std::cout << "Waiting for system...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto systems = mavsdk.systems();
        for (int i = 0; i < 50 && systems.empty(); ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            systems = mavsdk.systems();
        }
        if (systems.empty())
        {
            std::cerr << "No system found\n";
            running = false;
            return;
        }
        system = systems.front();
    }

    auto telemetry_plugin = mavsdk::Telemetry{system};

    telemetry_plugin.subscribe_attitude([&](mavsdk::Telemetry::Attitude att)
                                        {
        std::lock_guard<std::mutex> lk(telemetry.m);
        telemetry.roll = att.roll_rad;
        telemetry.pitch = att.pitch_rad;
        telemetry.yaw = att.yaw_rad; });

    telemetry_plugin.subscribe_position([&](mavsdk::Telemetry::Position pos)
                                        {
                                            std::lock_guard<std::mutex> lk(telemetry.m);
                                            telemetry.lat = pos.latitude_deg;
                                            telemetry.lon = pos.longitude_deg;
                                            telemetry.alt = pos.relative_altitude_m; // or pos.absolute_altitude_m depending on your setup
                                        });

    // keep thread alive
    while (running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// processing thread: inference + display
void processorThread()
{
    // load ONNX with OpenCV DNN
    cv::dnn::Net net = cv::dnn::readNet(MODEL_ONNX);
    // prefer backend/target (optional)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::namedWindow("Detection", cv::WINDOW_NORMAL);

    while (running)
    {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(qMutex);
            qCond.wait_for(lk, std::chrono::milliseconds(100), []
                           { return !frameQueue.empty() || !running; });
            if (!frameQueue.empty())
            {
                frame = frameQueue.back();
                frameQueue.clear(); // drop older frames
            }
        }
        if (frame.empty())
            continue;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 360));

        // ----- PREPROCESS for YOLO (depends on model) -----
        cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // parse outputs for YOLOv5/YOLOv8 style ONNX (single output with Nx85) adjust as model requires
        // Here we assume typical Nx(5+classes). We'll implement naive parsing and NMS.
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // reshape according to output
        for (auto &out : outputs)
        {
            const float *data = (float *)out.data;
            int rows = out.rows;
            int cols = out.cols;
            // If single row big matrix, adjust:
            if (rows == 1 && out.total() > 0)
            {
                // flatten
                int total = (int)out.total();
                int stride = cols; // safety
                rows = total / cols;
            }
            int num = out.total() / out.size[1]; // fallback
            int dims = out.size[1];

            // safe iteration via Mat iterator
            for (int i = 0; i < out.rows; ++i)
            {
                const float *row = out.ptr<float>(i);
                // expected format: [cx, cy, w, h, conf, cls0, cls1, ...]
                float box_conf = row[4];
                if (box_conf < 0.3f)
                    continue;
                // find best class
                float maxc = 0;
                int cls = -1;
                for (int c = 5; c < out.cols; ++c)
                {
                    if (row[c] > maxc)
                    {
                        maxc = row[c];
                        cls = c - 5;
                    }
                }
                float conf = box_conf * maxc;
                if (conf < 0.3f)
                    continue;
                float cx = row[0], cy = row[1], w = row[2], h = row[3];
                // coordinates are relative to 640x640 input; we used blobFromImage with size 640x640 but source resized to 640x360.
                // Need to map to resized image scale:
                float scale_x = (float)resized.cols / 640.0f;
                float scale_y = (float)resized.rows / 640.0f; // note: letterbox used? For robust mapping, use letterbox logic. This is simple approx.

                int left = int((cx - w / 2.0f) * scale_x);
                int top = int((cy - h / 2.0f) * scale_y);
                int width = int(w * scale_x);
                int height = int(h * scale_y);

                boxes.emplace_back(left, top, width, height);
                confidences.push_back(conf);
                classIds.push_back(cls);
            }
        }

        // NMS
        std::vector<int> idxs;
        cv::dnn::NMSBoxes(boxes, confidences, 0.3f, 0.45f, idxs);

        for (int i : idxs)
        {
            cv::Rect b = boxes[i];
            float conf = confidences[i];
            int cx = b.x + b.width / 2;
            int cy = b.y + b.height / 2;
            double lat, lon;
            if (pixelToGeo(cx, cy, lat, lon))
            {
                cv::rectangle(resized, b, cv::Scalar(0, 255, 0), 2);
                std::string txt = "cls:" + std::to_string(classIds[i]) + " " + std::to_string(conf);
                cv::putText(resized, txt, {b.x, b.y - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%.6f, %.6f", lat, lon);
                cv::putText(resized, buf, {b.x, b.y + b.height + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
            }
            else
            {
                cv::rectangle(resized, b, cv::Scalar(0, 0, 255), 2);
            }
        }

        cv::imshow("Detection", resized);
        if (cv::waitKey(1) == 'q')
        {
            running = false;
            break;
        }
    }
    cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
    // Optional: read connection url from args
    std::string mavlink_conn = "udp://:14540"; // change to your mvlink url or serial port

    std::thread t_reader(readerThread, VIDEO_PATH);
    std::thread t_mav(mavsdkThread, mavlink_conn);
    std::thread t_proc(processorThread);

    t_reader.join();
    t_mav.join();
    t_proc.join();
    return 0;
}
