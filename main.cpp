#include "openvino/openvino.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "format_reader/format_reader_ptr.h"
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <openvino/op/transpose.hpp>
#include <openvino/core/node.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include "inference_engine.hpp"

std::vector<std::string> readLabels(std::string &labelFilepath)
{
	std::vector<std::string> labels;
	std::string line;
	std::ifstream fp(labelFilepath);
	while (std::getline(fp, line))
	{
		labels.push_back(line);
	}
	return labels;
}

std::shared_ptr<ov::Model> preProcessIRModel(std::shared_ptr<ov::Model>& network)
{
    const ov::Layout tensor_layout{"NHWC"};
    ov::preprocess::PrePostProcessor ppp(network);
    // ppp.input().tensor().set_shape({1, 640, 640, 3});
    ov::preprocess::InputInfo& input_info = ppp.input();
    input_info.tensor().set_element_type(ov::element::f32).set_layout(tensor_layout);
    input_info.model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    ppp.output().postprocess().convert_layout({0, 2, 1});
    network = ppp.build();

    return network;

}

void postProcessYolov8OpenVINO(ov::Tensor& output, cv::Mat &original_img, const int &dimensions = 84)
{	
	float x_factor = original_img.size().width / 640;
	float y_factor = original_img.size().height / 640;

	std::string labelFilepath{"/media/ashray/D/Projects/x/coco80.txt"};
	std::vector<std::string> labels{readLabels(labelFilepath)};
	std::vector<cv::Rect> boxes;
	std::vector<float> max_class_score;
	std::vector<int> class_index;

	float *data = (float *)output.data();
    std::cout << *data << std::endl;

	for (int i = 0; i < output.get_size(); i += dimensions)
	{
		float x = (int)((data[i] - 0.5 * data[i + 2]) * x_factor);
		float y = (int)((data[i + 1] - 0.5 * data[i + 3]) * y_factor);
		float w = (int)(data[i + 2] * x_factor);
		float h = (int)(data[i + 3] * y_factor);

		cv::Rect bbox(x, y, w, h);
		boxes.push_back(bbox);

		std::vector<float> class_confidence;
		for (int j = i + 4; j <= 80 + i; j++)
		{
			class_confidence.push_back(data[j]);
		}

		auto result = std::max_element(class_confidence.begin(), class_confidence.end());

		int maxIndex = std::distance(class_confidence.begin(), std::max_element(class_confidence.begin(), class_confidence.end()));
		float maxConfidence = *result;
		class_index.push_back(maxIndex);
		max_class_score.push_back(maxConfidence);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, max_class_score, 0.4, 0.45, indices);

	cv::Mat pred_image = original_img.clone();
	cv::resize(pred_image, pred_image, cv::Size(604, 640));

	for (int i = 0; i < indices.size(); i++)
	{	
		const int class_id = class_index[indices[i]];
		const std::string class_name = labels[class_id];
		cv::rectangle(pred_image, boxes[indices[i]].tl(), boxes[indices[i]].br(), cv::Scalar(255, 0, 0), 2, cv::LINE_8);
		cv::putText(pred_image, class_name, cv::Point(boxes[indices[i]].x, boxes[indices[i]].y), 2, 0.5, cv::Scalar(0, 255, 0));
	}

	cv::imshow("Image", pred_image);
	cv::waitKey(0);
}

void postProcessYolov8OpenCV(cv::Mat &output, cv::Mat &original_img, const int &dimensions)
{	
	float x_factor = original_img.size().width / 640;
	float y_factor = original_img.size().height / 640;

	std::string labelFilepath{"/media/ashray/D/Projects/x/coco80.txt"};
	std::vector<std::string> labels{readLabels(labelFilepath)};
	std::vector<cv::Rect> boxes;
	std::vector<float> max_class_score;
	std::vector<int> class_index;

	float *data = (float *)output.data;
    std::cout << *data << std::endl;
	for (int i = 0; i < output.total(); i += dimensions)
	{
		float x = (int)((data[i] - 0.5 * data[i + 2]) * x_factor);
		float y = (int)((data[i + 1] - 0.5 * data[i + 3]) * y_factor);
		float w = (int)(data[i + 2] * x_factor);
		float h = (int)(data[i + 3] * y_factor);

		cv::Rect bbox(x, y, w, h);
		boxes.push_back(bbox);

		std::vector<float> class_confidence;
		for (int j = i + 4; j <= 80 + i; j++)
		{
			class_confidence.push_back(data[j]);
		}

		auto result = std::max_element(class_confidence.begin(), class_confidence.end());

		int maxIndex = std::distance(class_confidence.begin(), std::max_element(class_confidence.begin(), class_confidence.end()));
		float maxConfidence = *result;
		class_index.push_back(maxIndex);
		max_class_score.push_back(maxConfidence);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, max_class_score, 0.3, 0.45, indices);

	cv::Mat pred_image = original_img.clone();
	cv::resize(pred_image, pred_image, cv::Size(604, 640));

	for (int i = 0; i < indices.size(); i++)
	{	
		const int class_id = class_index[indices[i]];
		const std::string class_name = labels[class_id];
		cv::rectangle(pred_image, boxes[indices[i]].tl(), boxes[indices[i]].br(), cv::Scalar(255, 0, 0), 2, cv::LINE_8);
		cv::putText(pred_image, class_name, cv::Point(boxes[indices[i]].x, boxes[indices[i]].y), 2, 0.5, cv::Scalar(0, 255, 0));
	}

	cv::imshow("Image", pred_image);
	cv::waitKey(0);
}

void processOpenCV()
{
    std::cout << CV_VERSION << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromModelOptimizer("/media/ashray/D/yolov8n_openvino_int8_model/yolov8n.xml", "/media/ashray/D/yolov8n_openvino_int8_model/yolov8n.bin");
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	cv::Mat img = cv::imread("/media/ashray/D/image5.jpg");

	cv::Mat blob = cv::dnn::blobFromImage(img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	std::vector<cv::String> layer_names = net.getLayerNames();

	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	cv::Mat output;
	cv::transposeND(outputs[0], {0, 2, 1}, output);

	const int dimensions = outputs[0].size[1];
	postProcessYolov8OpenCV(output, img, dimensions);
}

int main()
{	
	// std::cout << "OpenVINO version: " << InferenceEngine::GetInferenceEngineVersion()->buildNumber << std::endl;
	// std::cout << "OpenVINO version: " << InferenceEngine::GetInferenceEngineVersion()->description << std::endl;

	std::cout << ov::get_openvino_version() << std::endl;
	ov::Core core;
    const std::string model_path = "/media/ashray/D/yolov8n_openvino_int8_model/yolov8n.xml";
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    std::cout << "Model reading completed" << std::endl;

    // model->reshape({1, 3, 640, 640});
    // printInputAndOutputscout(*model);

    const std::string image_path = "/media/ashray/D/image9.jpg";
    cv::Mat image = cv::imread(image_path);
	cv::Mat resized_image;
	cv::resize(image, resized_image, cv::Size(640, 640));
	resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
    // cv::imwrite("/media/ashray/D/Projects/openvino_samples/resize/image.jpg", resize_image);
    // FormatReader::ReaderPtr reader(resize_image_path.c_str());
    // if (reader.get() == nullptr)
    // {
    //     std::stringstream ss;
    //     ss << "Image " + image_path + " cannot be read!";
    //     throw std::logic_error(ss.str());
    // }

    ov::element::Type input_type = ov::element::f32;
    ov::Shape input_shape = {1, 640, 640, 3};
    // std::shared_ptr<unsigned char> input_data = reader->getData();


    // ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());
	ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, resized_image.data);
    model = preProcessIRModel(model);

    const std::string device_name = "CPU";
    ov::CompiledModel compiled_model = core.compile_model(model, device_name);
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    std::cout << output_tensor.get_size() << std::endl;

    postProcessYolov8OpenVINO(output_tensor, image);
    // processOpenCV();
    return 0;
}