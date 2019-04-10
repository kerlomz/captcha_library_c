// CaptchaLibrary.cpp : Defines the exported functions for the DLL application.
//
#pragma warning(disable:4996)
#include "stdafx.h"

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "yaml-cpp/yaml.h"
#include "base64.h"
#include <typeinfo>


using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

static const string *Numeric;
static const string *Alphanumeric;
static const string *AlphabetLower;
static const string *AlphabetUpper;
static const string *AlphanumericUpper;
static const string *AlphanumericLower;

static const string *charset_map = new string;
static Session* session;
static YAML::Node config = YAML::LoadFile("./model.yaml");
static const std::vector<int> resize = config["Pretreatment"]["Resize"].as<std::vector<int>>();
static const std::vector<string> exclude = config["Model"]["CharExclude"].as<std::vector<string>>();
static const std::string model_name = config["Model"]["ModelName"].as<std::string>();
static YAML::Node _charset = config["Model"]["CharSet"];
static int charset_size = 0;

int InitSession()
{

	Numeric = new string[11] { "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
	Alphanumeric = new string[63]{
		"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
		"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
	};
	AlphabetLower = new string[27]{ "", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" };
	AlphabetUpper = new string[27]{ "", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
	AlphanumericUpper = new string[37]{ "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
	AlphanumericLower = new string[37]{ "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" };

	Status status = NewSession(SessionOptions(), &session);


	if (!config["Model"]) {
		return -1;
	}

	if (_charset.size() == 0) {
		const std::string charset = _charset.as<std::string>();
		if (charset.compare("NUMERIC") == 0) {
			charset_map = Numeric;
			charset_size = 11;
		}
		else if (charset.compare("ALPHANUMERIC") == 0) {
			charset_map = Alphanumeric;
			charset_size = 63;
		}
		else if (charset.compare("ALPHABET_LOWER") == 0) {
			charset_map = AlphabetLower;
			charset_size = 27;
		}
		else if (charset.compare("ALPHABET_UPPER") == 0) {
			charset_map = AlphabetUpper;
			charset_size = 27;
		}
		else if (charset.compare("ALPHANUMERIC_UPPER") == 0) {
			charset_map = AlphanumericUpper;
			charset_size = 37;
		}
		else if (charset.compare("ALPHANUMERIC_LOWER") == 0) {
			charset_map = AlphanumericLower;
			charset_size = 37;
		}
		else {
			charset_map = AlphanumericLower;
			charset_size = 37;
		}
	}
	else if (_charset.size() > 1) {
		std::vector<string> charset = _charset.as<std::vector<string>>();
		int size = charset.size() + 1;
		string* x = new string[size];

		vector<string>::const_iterator i = charset.begin();
		int n = 1;
		x[0] = "";
		while (i != charset.end())
		{
			x[n] = (*i);
			n++;
			i++;
		}

		charset_map = x;
		charset_size = size;
	}
	else {
		charset_map = AlphanumericLower;
	}


	string model_path = "./" + model_name + ".pb";
	GraphDef graphdef;

	Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!status_load.ok()) {
		std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
		std::cout << status_load.ToString() << "\n";
		return -1;
	}
	Status status_create = session->Create(graphdef);
	if (!status_create.ok()) {
		std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
		return -1;
	}
	cout << "Session successfully created." << endl;

	return 0;
}

static Status ReadBytesString(tensorflow::Env* env, string buffer, Tensor* output) {

	output->scalar<string>()() = tensorflow::StringPiece(buffer).ToString();
	return Status::OK();
}

Status ReadTensorFromImageBytesString(string buffer, const int input_height, const int input_width, const float input_mean, const float input_std, std::vector<Tensor>* out_tensors) {

	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
	TF_RETURN_IF_ERROR(
		ReadBytesString(tensorflow::Env::Default(), buffer, &input));

	auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{"input", input},
	};

	const int wanted_channels = 1;
	tensorflow::Output image_reader = DecodePng(root.WithOpName("png_reader"), file_reader, DecodePng::Channels(wanted_channels));

	auto _dims_expander = ExpandDims(root.WithOpName("expand"), image_reader, 0);
	auto _resize = ResizeArea(root, _dims_expander, Const(root, { input_height, input_width }));
	Output float_caster = Cast(root.WithOpName("float_caster"), _resize, tensorflow::DT_FLOAT);
	auto _reshape = Reshape(root, float_caster, Const(root, { input_height, input_width }));
	auto transpose = Transpose(root, _reshape, Const(root, { 1, 0 }));
	auto resized = Reshape(root, transpose, Const(root, { input_width, input_height, 1 }));
	auto dims_expander = ExpandDims(root.WithOpName("expand"), resized, 0);

	float input_max = 255;
	Div(root.WithOpName("div"), dims_expander, input_max);

	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { "div" }, {}, out_tensors));
	return Status::OK();
}


static Status ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output) {

	string contents;
	tensorflow::uint64 file_size = 0;
	TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
	contents.resize(file_size);

	std::unique_ptr<tensorflow::RandomAccessFile> file;
	TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

	tensorflow::StringPiece data;
	TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
	if (data.size() != file_size) {
		return tensorflow::errors::DataLoss("Truncated read of '", filename, "' expected ", file_size, " got ", data.size());
	}
	output->scalar<string>()() = data.ToString();
	return Status::OK();
}


static Status ReadTensorFromImageFile(const string& file_name, const int input_height, const int input_width, const float input_mean, const float input_std, std::vector<Tensor>* out_tensors) {
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
	TF_RETURN_IF_ERROR(
		ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

	auto file_reader = Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

	std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
		{"input", input},
	};

	const int wanted_channels = 1;
	tensorflow::Output image_reader;
	if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
		image_reader =
			Squeeze(root.WithOpName("squeeze_first_dim"),
				DecodeGif(root.WithOpName("gif_reader"), file_reader));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
		image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
	}
	else {
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}
	auto _dims_expander = ExpandDims(root.WithOpName("expand"), image_reader, 0);
	auto _resize = ResizeArea(root, _dims_expander, Const(root, { input_height, input_width }));
	Output float_caster = Cast(root.WithOpName("float_caster"), _resize, tensorflow::DT_FLOAT);
	auto _reshape = Reshape(root, float_caster, Const(root, { input_height, input_width }));
	auto transpose = Transpose(root, _reshape, Const(root, { 1, 0 }));
	auto resized = Reshape(root, transpose, Const(root, { input_width, input_height, 1 }));
	auto dims_expander = ExpandDims(root.WithOpName("expand"), resized, 0);

	float input_max = 255;
	Div(root.WithOpName("div"), dims_expander, input_max);

	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({ inputs }, { "div" }, {}, out_tensors));
	return Status::OK();
}

char * Decode(vector<tensorflow::Tensor> outputs) {
	Tensor t = outputs[0];
	string result = "";
	auto tmap = t.tensor<int64, 2>();
	int output_dim = t.shape().dim_size(1);
	for (int j = 0; j < output_dim; j++)
	{
		int index = tmap(0, j);
		if (index >= charset_size) {
			std::cout << "ERROR: CHARSET error, index[" << index << "] > charset size[" << charset_size << "]..." << std::endl;
			return NULL;
		}
		result += charset_map[tmap(0, j)];
	}
	int len = result.length();
	char *p = new char[len + 1];
	p[len] = 0;
	strcpy(p, result.c_str());
	return p;
}


char * PredictFile(char *image_path) {

	int input_height = resize[1];
	int input_width = resize[0];
	int input_mean = 0;
	int input_std = 1;
	std::vector<Tensor> resized_tensors;

	Status read_tensor_status = ReadTensorFromImageFile(image_path, input_height, input_width, input_mean, input_std, &resized_tensors);
	if (!read_tensor_status.ok()) {
		LOG(ERROR) << read_tensor_status;
		cout << "resing error" << endl;
		return NULL;
	}

	const Tensor& resized_tensor = resized_tensors[0];

	vector<tensorflow::Tensor> outputs;
	string output_node = "dense_decoded";
	Status status_run = session->Run({ {"input", resized_tensor} }, { output_node }, {}, &outputs);

	if (!status_run.ok()) {
		std::cout << "ERROR: RUN failed..." << std::endl;
		std::cout << status_run.ToString() << "\n";
		return NULL;
	}

	return Decode(outputs);
}

char *PredictBase64(char *img_base64) {

	int input_height = resize[1];
	int input_width = resize[0];
	int input_mean = 0;
	int input_std = 1;
	std::vector<Tensor> resized_tensors;

	string image_bytes = base64_decode(img_base64);
	
	Status read_tensor_status = ReadTensorFromImageBytesString(image_bytes, input_height, input_width, input_mean, input_std, &resized_tensors);
	if (!read_tensor_status.ok()) {
		LOG(ERROR) << read_tensor_status;
		cout << "resing error" << endl;
		return NULL;
	}

	const Tensor& resized_tensor = resized_tensors[0];

	vector<tensorflow::Tensor> outputs;
	string output_node = "dense_decoded";
	Status status_run = session->Run({ {"input", resized_tensor} }, { output_node }, {}, &outputs);

	if (!status_run.ok()) {
		std::cout << "ERROR: RUN failed..." << std::endl;
		std::cout << status_run.ToString() << "\n";
		return NULL;
	}

	return Decode(outputs);
}

char *PredictBinData(const unsigned char *img_data, const unsigned int& data_size)
{
	int input_height = resize[1];
	int input_width = resize[0];
	int input_mean = 0;
	int input_std = 1;
	std::vector<Tensor> resized_tensors;

	string image_bytes;
	for (unsigned int i = 0; i < data_size; i++)
		image_bytes.push_back(img_data[i]);

	Status read_tensor_status = ReadTensorFromImageBytesString(image_bytes, input_height, input_width, input_mean, input_std, &resized_tensors);
	if (!read_tensor_status.ok())
	{
		LOG(ERROR) << read_tensor_status;
		cout << "resing error" << endl;
		return NULL;
	}

	const Tensor& resized_tensor = resized_tensors[0];

	vector<tensorflow::Tensor> outputs;
	string output_node = "dense_decoded";
	Status status_run = session->Run({ { "input", resized_tensor } }, { output_node }, {}, &outputs);

	if (!status_run.ok()) {
		std::cout << "ERROR: RUN failed..." << std::endl;
		std::cout << status_run.ToString() << "\n";
		return NULL;
	}

	return Decode(outputs);
}

int  FreePredictData(char *& pPredictData)
{
	if (pPredictData != NULL)
	{
		delete[]pPredictData;
		pPredictData = NULL;
		return 0;
	}
	else
	{
		return -1;
	}
}


void FreeSession() {
	session->Close();
}