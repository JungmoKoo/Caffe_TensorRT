#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <NvInfer.h>
#include <NvCaffeParser.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;
using namespace cv;

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define LOG_GIE "[GIE]  "

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
static const int CHANNEL_SIZE = 3;
const int BATCH_SIZE = 1;

bool mEnableFP16=false;
bool mOverride16=false;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void WrapInputLayer(std::vector<std::vector<cv::Mat> >& input_channels, float* buffer)
{
    float* input_data = buffer;

    for (int n = 0; n < 1; ++n)
    {
        input_channels.push_back(std::vector<cv::Mat>());
        for (int i = 0; i < CHANNEL_SIZE; ++i)
        {
            cv::Mat channel(INPUT_H, INPUT_W, CV_32FC1, input_data);
            input_channels[n].push_back(channel);
            input_data += INPUT_H * INPUT_W;
        }
    }
}

#define CHECK(status)                       \
{                                           \
    if (status != 0)                        \
{                                           \
    std::cout << "Cuda failure: " << status;\
    abort();                                \
    }                                       \
    }


// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;


void Preprocess(const cv::Mat& img, std::vector<std::vector<cv::Mat>> &input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    int num_channels_ = input_channels[0].size();
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Size input_geometry = cv::Size(input_channels[0][0].cols, input_channels[0][0].rows);

    cv::Mat sample_resized;
    /*preproc-resample */
    if (sample.size() != input_geometry)
        cv::resize(sample, sample_resized, input_geometry);
    else
        sample_resized = sample;
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    /* END */
    /* preproc-normalize */
    cv::Mat sample_normalized;
    sample_float.copyTo(sample_normalized);

    for (int n = 0; n < BATCH_SIZE; ++n)
        cv::split(sample_normalized, input_channels[n]);
}



int main(int argc, char** argv)
{
    ::google::InitGoogleLogging(argv[0]);

    string model_file   = "../models/google/deploy.prototxt";
    string trained_file = "../models/google/bvlc_googlenet.caffemodel";
//    string mean_file    = "fake";
    string label_file   = "../models/google/imagenet1000.txt";


    /* Load labels. */
    std::vector<string> labels_;
    std::ifstream labels(label_file.c_str());
    //CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));


    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
    printf(LOG_GIE "platform %s FP16 support.\n", mEnableFP16 ? "has" : "does not have");
    printf(LOG_GIE "loading %s %s\n", model_file.c_str(), trained_file.c_str());

    nvinfer1::DataType modelDataType = mEnableFP16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
    const IBlobNameToTensor *blobNameToTensor = parser->parse(model_file.c_str(),		// caffe deploy file
                                                              trained_file.c_str(),		// caffe model file
                                                              *network,		// network definition that the parser will populate
                                                              modelDataType);

    assert(blobNameToTensor != nullptr);
    // the caffe file has no notion of outputs
    // so we need to manually say which tensors the engine should generate
    network->markOutput(*blobNameToTensor->find(OUTPUT_BLOB_NAME));
    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(16 << 20);//WORKSPACE_SIZE);

    // set up the network for paired-fp16 format
    if(mEnableFP16)
        builder->setHalf2Mode(true);

    // Eliminate the side-effect from the delay of GPU frequency boost
    builder->setMinFindIterations(3);
    builder->setAverageFindIterations(2);

    //build
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    IExecutionContext *context = engine->createExecutionContext();

    // run inference
    float prob[OUTPUT_SIZE];

    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * CHANNEL_SIZE * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;

    cv::Mat frame;
    cv::VideoCapture cap(0);
    void** mInputCPU= (void**)malloc(2*sizeof(void*));
    cudaHostAlloc((void**)&mInputCPU[0], CHANNEL_SIZE*INPUT_H*INPUT_W*sizeof(float), cudaHostAllocDefault);
    clock_t start_point, end_point;
    while(1)
    {
        start_point = clock();
        cap >> frame;

        std::vector<std::vector<cv::Mat> > input_channels;
        WrapInputLayer(input_channels, (float*)mInputCPU[0]);
        Preprocess(frame, input_channels);

        CHECK(cudaStreamCreate(&stream));
        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        CHECK(cudaMemcpyAsync(buffers[inputIndex], mInputCPU[0], BATCH_SIZE * CHANNEL_SIZE * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueue(BATCH_SIZE, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(prob, buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        end_point = clock();
        printf("frame rate : %f Hz\n", 1 / ((double)(end_point - start_point) / CLOCKS_PER_SEC));
        double delta_t_oneframe = (double)(end_point - start_point) / CLOCKS_PER_SEC;

        std::vector<float> probs;

        for (int n = 0; n < OUTPUT_SIZE; n++)
            probs.push_back(prob[n]);

        std::vector<std::pair<float, int> > pairs;
        for (size_t i = 0; i < probs.size(); ++i)
            pairs.push_back(std::make_pair(probs[i], i));
        std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(), PairCompare);

        std::vector<int> result;
        for (int i = 0; i < 5; ++i)
        {
            result.push_back(pairs[i].second);
            cout << prob[pairs[i].second] << " - " << labels_[pairs[i].second] << endl;
        }
        cout << endl;

        input_channels.clear();
        probs.clear();
        pairs.clear();
        result.clear();

        cv::imshow("img", frame);
        cv::waitKey(1);
    }

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    // destroy the engine
    context->destroy();
    engine->destroy();

    return 0;
}
