// yolo_onnx_demo.cpp
// Build: see CMakeLists below
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cmath>

// ----------------- CONFIG -----------------
const std::string RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
const std::string ONNX_PATH = R"(D:\github lite\cv_projects\Identifying and locating flags from aerial photography\models\trained_models\fuck any one8\weights\best.onnx)";

// camera intrinsics
const cv::Mat K = (cv::Mat_<float>(3,3) << 10600.0f, 0.0f, 1920.0f,
                                           0.0f, 10600.0f, 1080.0f,
                                           0.0f, 0.0f, 1.0f);
const cv::Mat DIST = cv::Mat::zeros(1,5,CV_32F);

// drone geo
const double DRONE_LAT = 30.1;
const double DRONE_LON = 30.6;
const double DRONE_ALT = 70.0; // meters

// model input
const int MODEL_W = 608;
const int MODEL_H = 608;
const float SCORE_THRESH = 0.4f;
const float NMS_THRESH = 0.45f;

// queue for frames
std::queue<cv::Mat> frame_queue;
std::mutex q_mutex;
std::condition_variable q_cond;
bool stop_flag = false;

// simple conversion: meters -> lat/lon delta (approx)
constexpr double EARTH_RADIUS_M = 6378137.0;
void meters_to_latlon(double dx, double dy, double lat0, double lon0, double &out_lat, double &out_lon) {
    // dy: north (positive), dx: east (positive)
    out_lat = lat0 + (dy / EARTH_RADIUS_M) * (180.0 / M_PI);
    out_lon = lon0 + (dx / (EARTH_RADIUS_M * std::cos(lat0 * M_PI / 180.0))) * (180.0 / M_PI);
}

// undistort pixel -> normalized ray
cv::Vec3f undistort_and_ray(int cx, int cy) {
    std::vector<cv::Point2f> pts = {cv::Point2f((float)cx, (float)cy)};
    std::vector<cv::Point2f> und;
    cv::undistortPoints(pts, und, K, DIST);
    // und points are normalized (x, y), assume z=1
    return cv::Vec3f(und[0].x, und[0].y, 1.0f);
}

// pixel -> lat lon using pinhole approximation (flat ground)
void pixel_to_geo(int cx, int cy, double &lat, double &lon) {
    cv::Vec3f ray = undistort_and_ray(cx, cy);
    // scale so z reaches altitude plane
    double scale = DRONE_ALT / ray[2];
    double dx = ray[0] * scale; // in camera units; assume units = meters at focal scale (approx)
    double dy = ray[1] * scale;
    // here dx,dy are in same units as altitude assumption (meters approx)
    // note: depending on camera intrinsics units/focal length scale, dx/dy approximate meters on ground
    meters_to_latlon(dx, dy, DRONE_LAT, DRONE_LON, lat, lon);
}

// capture thread
void capture_thread_func(const std::string &url) {
    cv::VideoCapture cap(url);
    if(!cap.isOpened()){
        std::cerr << "ERROR: cannot open video: " << url << std::endl;
        stop_flag = true;
        q_cond.notify_all();
        return;
    }
    while(!stop_flag) {
        cv::Mat frame;
        if(!cap.read(frame)) {
            std::cerr << "Stream ended or read error\n";
            break;
        }
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            if (frame_queue.size() < 5) {
                frame_queue.push(frame);
                q_cond.notify_one();
            } else {
                // drop frame if queue full
            }
        }
        // slight sleep to avoid busy loop
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cap.release();
    stop_flag = true;
    q_cond.notify_all();
}

// helper: print Mat shape
std::string shape_to_string(const cv::Mat &m) {
    std::ostringstream oss;
    oss << "(";
    for (int i=0;i<m.dims;i++){
        if (i) oss << ",";
        oss << m.size[i];
    }
    oss << ")";
    return oss.str();
}

int main(){
    // load network
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(ONNX_PATH);
    } catch (std::exception &e) {
        std::cerr << "Failed to load ONNX: " << e.what() << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Loaded ONNX model: " << ONNX_PATH << std::endl;

    // start capture thread
    std::thread cap_thread(capture_thread_func, RTSP_URL);

    cv::namedWindow("Det", cv::WINDOW_NORMAL);
    cv::resizeWindow("Det", 1000, 600);

    while(!stop_flag) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            q_cond.wait_for(lock, std::chrono::milliseconds(200), []{ return !frame_queue.empty() || stop_flag; });
            if(frame_queue.empty()) {
                if(stop_flag) break;
                continue;
            }
            frame = frame_queue.front();
            frame_queue.pop();
        }

        // resize for display/inference
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(MODEL_W, MODEL_H)); // resize to model input
        cv::Mat blob;
        // Assumptions: model expects float32 0..1 and RGB
        cv::dnn::blobFromImage(resized, blob, 1.0/255.0, cv::Size(MODEL_W, MODEL_H), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        try {
            net.forward(outputs, net.getUnconnectedOutLayersNames());
        } catch (std::exception &e) {
            std::cerr << "Forward failed: " << e.what() << std::endl;
            break;
        }

        // debug: print shapes
        // Typically ultralytics ONNX may return one Mat; print shapes
        std::cout << "Outputs count: " << outputs.size() << std::endl;
        for (size_t i=0;i<outputs.size();++i) {
            std::cout << "Output["<<i<<"] shape: " << shape_to_string(outputs[i]) << std::endl;
        }

        // --- parse detections ---
        // We try common layouts:
        // 1) (1, N, 6) -> [x1, y1, x2, y2, conf, class]
        // 2) (1, 5, M) -> per-ultralytics raw (cx, cy, w, h, conf) with M anchors; need reshape
        // 3) (1, num_preds, 85) -> yolov5 style (x, y, w, h, conf, cls0..clsN)
        // We'll try to handle case (1) and (3) easily. If your ONNX is different, inspect printed shapes and adapt.

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        // Try typical case: outputs.size()==1 and dims == 3: (1, num_preds, 6 or 85)
        if(outputs.size() == 1 && outputs[0].dims == 3) {
            cv::Mat out = outputs[0]; // shape e.g. [1, num, 6] or [1, num, 85]
            int batch = out.size[0];
            int num = out.size[1];
            int elem = out.size[2];
            // convert to float pointer
            float *data = (float*)out.data;
            // iterate predictions
            for (int i = 0; i < num; ++i) {
                float x = data[i*elem + 0];
                float y = data[i*elem + 1];
                float w = data[i*elem + 2];
                float h = data[i*elem + 3];
                float conf = data[i*elem + 4];
                // if elem > 5 -> class scores follow
                int cls = 0;
                float cls_conf = 0.0f;
                if (elem > 5) {
                    // find best class
                    int best_j = 5;
                    cls_conf = data[i*elem + 5];
                    cls = 0;
                    for (int j = 6; j < elem; ++j) {
                        float c = data[i*elem + j];
                        if (c > cls_conf) { cls_conf = c; cls = j-5; }
                    }
                }
                float score = conf * std::max(1.0f, cls_conf);
                if (score < SCORE_THRESH) continue;

                // if the model outputs center x,y,w,h normalized to input size, convert to x1,y1
                // We try to detect whether x,y are center or top-left by checking ranges (common: normalized 0..1 or pixel coords)
                // assume normalized center
                float cx = x, cy = y;
                float bw = w, bh = h;
                // if values likely in [0,1], scale to image
                if (cx <= 1.5f && cy <= 1.5f && bw <= 1.5f && bh <= 1.5f) {
                    cx *= MODEL_W; cy *= MODEL_H; bw *= MODEL_W; bh *= MODEL_H;
                }
                int x1 = int(cx - bw/2.0f);
                int y1 = int(cy - bh/2.0f);
                int x2 = int(cx + bw/2.0f);
                int y2 = int(cy + bh/2.0f);
                // clamp
                x1 = std::max(0, std::min(x1, MODEL_W-1));
                y1 = std::max(0, std::min(y1, MODEL_H-1));
                x2 = std::max(0, std::min(x2, MODEL_W-1));
                y2 = std::max(0, std::min(y2, MODEL_H-1));

                boxes.emplace_back(x1, y1, x2-x1, y2-y1);
                scores.push_back(score);
                classIds.push_back(cls);
            }
        } else {
            // fallback: try outputs that are (1,5,M) like ultralytics raw -> convert by reshaping
            for (auto &m : outputs) {
                // print already done above
            }
            // If reach here and nothing parsed, skip drawing and continue
            if (boxes.empty()) {
                cv::imshow("Det", resized);
                if (cv::waitKey(1) == 'q') break;
                continue;
            }
        }

        // apply NMS
        std::vector<int> idxs;
        cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESH, NMS_THRESH, idxs);

        // draw on display image (we used resized for inference)
        for (int id : idxs) {
            cv::Rect b = boxes[id];
            int cx = b.x + b.width/2;
            int cy = b.y + b.height/2;
            double lat, lon;
            pixel_to_geo(cx, cy, lat, lon);
            std::string label = std::to_string(classIds[id]);
            char buf[200];
            std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), scores[id]);
            cv::rectangle(resized, b, cv::Scalar(0,255,0), 2);
            cv::putText(resized, buf, cv::Point(b.x, b.y-8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
            char geo[200];
            std::snprintf(geo, sizeof(geo), "%.5f, %.5f", lat, lon);
            cv::putText(resized, geo, cv::Point(b.x, b.y + b.height + 15), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,0), 1);
        }

        cv::imshow("Det", resized);
        if (cv::waitKey(1) == 'q') {
            stop_flag = true;
            break;
        }
    }

    if(cap_thread.joinable()) cap_thread.join();
    cv::destroyAllWindows();
    return 0;
}
