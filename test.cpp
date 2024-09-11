#include <assert.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "utils.cpp"
using std::cout;
using std::endl;

using namespace cv;
using namespace std;

class depthAnything
{
public:
    depthAnything(const wchar_t* onnx_model_path);
    std::vector<float> predict(std::vector<float>& input_data, int batch_size = 1, int index = 0);
    cv::Mat predict(cv::Mat& input_tensor, int batch_size = 1, int index = 0);
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*>input_node_names = {"image"};
    std::vector<const char*>output_node_names = {"depth"};
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
};
depthAnything::depthAnything(const wchar_t* onnx_model_path) :session(nullptr), env(nullptr)
{
    // init env
    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "depthAnything_mono");
    // init session options
    Ort::SessionOptions session_options;
    // session_options.SetInterOpNumThreads(1);
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // create session and load to memory
    this->session = Ort::Session(env, onnx_model_path, session_options);
    //输入输出节点数量和名称
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    for (int i = 0; i < num_input_nodes; i++)
    {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->input_node_dims = tensor_info.GetShape();
    }
    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->output_node_dims = tensor_info.GetShape();
    }
}

std::vector<float> depthAnything::predict(std::vector<float>& input_tensor_values, int batch_size, int index)
{
    this->input_node_dims[0] = batch_size;
    this->output_node_dims[0] = batch_size;
    float* floatarr = nullptr;
    try
    {
        std::vector<const char*>output_node_names;
        if (index != -1)
        {
            output_node_names = { this->output_node_names[index] };
        }
        else
        {
            output_node_names = this->output_node_names;
        }
        this->input_node_dims[0] = batch_size;
        auto input_tensor_size = input_tensor_values.size();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        floatarr = output_tensors[0].GetTensorMutableData<float>();
    }
    catch (Ort::Exception& e)
    {
        throw e;
    }
    int64_t output_tensor_size = 1;
    for (auto& it : this->output_node_dims)
    {
        output_tensor_size *= it;
    }
    std::vector<float>results(output_tensor_size);
    for (unsigned i = 0; i < output_tensor_size; i++)
    {
        results[i] = floatarr[i];
    }
    return results;
}
cv::Mat depthAnything::predict(cv::Mat& input_tensor, int batch_size, int index)
{
    int input_tensor_size = input_tensor.cols * input_tensor.rows * 3;
    std::size_t counter = 0;
    std::vector<float>input_data(input_tensor_size);
    std::vector<float>output_data;
    try
    {
        for (unsigned k = 0; k < 3; k++)
        {
            for (unsigned i = 0; i < input_tensor.rows; i++)
            {
                for (unsigned j = 0; j < input_tensor.cols; j++)
                {
                    input_data[counter++] = static_cast<float>(input_tensor.at<cv::Vec3b>(i, j)[k]) / 255.0;
                }
            }
        }
    }
    catch (cv::Exception& e)
    {
        printf(e.what());
    }
    try
    {
        output_data = this->predict(input_data);
    }
    catch (Ort::Exception& e)
    {
        throw e;
    }
    cv::Mat output_tensor(output_data);
    output_tensor =output_tensor.reshape(1, {518, 518});
    double minVal, maxVal;
    cv::minMaxLoc(output_tensor, &minVal, &maxVal);
    output_tensor.convertTo(output_tensor, CV_32F);
    if (minVal != maxVal) {
        output_tensor = (output_tensor - minVal) / (maxVal - minVal);
        
    }
    output_tensor *= 255.0;
    output_tensor.convertTo(output_tensor, CV_8UC1);
    cv::applyColorMap(output_tensor, output_tensor, cv::COLORMAP_JET);
    return output_tensor;
}
int main(int argc, char* argv[])
{
    // const wchar_t* model_path = L"model/simvit.onnx";
    // depthAnything model(model_path);
    // cv::Mat image = cv::imread("inference/DSC_0410.jpg");
    // auto ori_h = image.cols;
    // auto ori_w = image.rows;
    // cv::imshow("image", image);
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // cv::resize(image, image, {518, 518}, 0.0, 0.0, cv::INTER_CUBIC);
    // auto result = model.predict(image);
    // cv::resize(result, result, {ori_h, ori_w}, 0.0, 0.0, cv::INTER_CUBIC);
    // cv::imwrite("da.png",result);
    // // cv::imshow("result", result);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    const wchar_t* model_path = L"model/simvit.onnx";
    depthAnything model(model_path);
    cv::Mat image = cv::imread("inference/classroom.jpg");
    auto ori_h = image.cols;
    auto ori_w = image.rows;
    string kWinName = "Deep learning depth estimation DepthAnything in OpenCV";
    VideoCapture capture(1);
    Mat frame, temp;
    while(true){
    	capture >> frame;
    	ori_h = frame.cols;
    	ori_w = frame.rows;
    	resize(frame, temp, Size(518, 518), INTER_LINEAR);
    	cv::cvtColor(temp, temp, cv::COLOR_BGR2RGB);
    	Mat depthMap = model.predict(temp);
    	cv::resize(depthMap, depthMap, {ori_h, ori_w}, 0.0, 0.0, cv::INTER_CUBIC);
    	Mat res = viewer({frame, depthMap});
    	if(waitKey(10) == 'q'){
    		capture.release();
    		break;
    	}
    	imshow(kWinName, res);
    }
}

