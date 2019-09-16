#include <memory>
#include <vector>
#include <sstream>
#include <iostream>
#include <experimental/filesystem>

//#include <opencv2/opencv.hpp>

#include <openpose/headers.hpp>
#include <darknet.h>

#include "detct_objDetect.h"

const int MAX_CHAR_LEN             = 512;
const std::string CFG_FILE         = "/home/musabbir/workspace/poseDetection/darknet/cfg/yolov3.cfg";
const std::string WEIGHT_FILE      = "/home/musabbir/workspace/poseDetection/darknet/yolov3.weights";
const std::string IMAGE_FILE       = "/home/musabbir/workspace/poseDetection/images/crawl0.jpg";
const std::string CLASS_NAMES_FILE = "/home/musabbir/workspace/poseDetection/darknet/data/coco.names";
const std::string VIDEO_PATH       = "/home/musabbir/workspace/poseDetection/deep_sort/resources/videos/img1/";
const std::string DETECTIONS_FILE  = "/home/musabbir/workspace/poseDetection/deep_sort/resources/videos/det/det.txt";
const op::PoseModel POSE_MODEL     = op::PoseModel::BODY_25;
const char delimiter               = '.';
const std::string FILE_EXTENSSION  = ".jpeg";
const float THRESHOLD              = 0.5;
const float HIER_THRESHOLD         = 0.5;
const int BBOX_PARAMETERS          = 4;


std::unique_ptr<network> configureNetwork()
{
	char cfgFileTemp[MAX_CHAR_LEN];
	char weightFileTemp[MAX_CHAR_LEN];
	return std::unique_ptr<network> (load_network(strcpy(cfgFileTemp, CFG_FILE.c_str()),
											  strcpy(weightFileTemp, WEIGHT_FILE.c_str()),
											  0));

	//! @TODO - raise exception
}

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

cv::Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    cv::Mat m = cv::cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image loadImage(const std::string& imageFilePath)
{
	char imageFileTemp[MAX_CHAR_LEN];
	strcpy(imageFileTemp, imageFilePath.c_str());
	return load_image_color(imageFileTemp, 0, 0);
}

detection* detectObjectsInImage(int& nBox, std::unique_ptr<network>& net,
					     	    image & im)
{
	image sized = letterbox_image(im, net->w, net->h);
	float* netDataNum = sized.data;
	if(!netDataNum) {
		std::cout << "netDataNum is null \n";
	}

	set_batch_network(net.get(), 1); //! 1 = batch size
	network_predict(net.get(), netDataNum);

	return get_network_boxes(net.get(), im.w, im.h, THRESHOLD,
							 HIER_THRESHOLD, 0, 1, &nBox);


}

std::array<double, BBOX_PARAMETERS> getPersonBbox(image & im, box & objBbox) {
	//! add max/min logic
	const double width   = objBbox.w * im.w;
	const double height  = objBbox.h * im.h;
	const double dx      = (objBbox.x * im.w - 0.5 * width) < 0. ? 0.
							: (objBbox.x * im.w - 0.5 * width);
	const double dy      = (objBbox.y * im.h - 0.5 * height) < 0. ? 0.
						 	: (objBbox.y * im.h - 0.5 * height);

	return std::array<double, BBOX_PARAMETERS> {{dx, dy, width, height}};
}

std::vector<std::array<double, BBOX_PARAMETERS>> getPersonBboxes(image im, detection *dets,
											       int numBoxes)
{
	std::vector<std::array<double, BBOX_PARAMETERS>> boundingBoxes;

    for(int i = 0; i < numBoxes; ++i) {
		if (dets[i].prob[0] > THRESHOLD) { //! prob[0] is 'person'
			//! guaranteed to be a person
			box objBoundingBox = dets[i].bbox;

        	std::array<double, BBOX_PARAMETERS> boundingBox = getPersonBbox(im, objBoundingBox);
        	boundingBoxes.push_back(boundingBox);
		}
    }

    return boundingBoxes;
}

std::vector<cv::Mat> getImageLabels(image im, detection *dets, int numBoxes)
{
	char classNamesFileTemp[MAX_CHAR_LEN];
	strcpy(classNamesFileTemp, CLASS_NAMES_FILE.c_str());
	std::vector<cv::Mat> images;

    for(int i = 0; i < numBoxes; ++i) {
		if (dets[i].prob[0] > THRESHOLD) { //! prob[0] is 'person'
			//! guaranteed to be a person
			box objBoundingBox = dets[i].bbox;
			std::array<double, BBOX_PARAMETERS> boundingBox = getPersonBbox(im, objBoundingBox);

			image objCroppedImage = crop_image(im, boundingBox[0], boundingBox[1],
				                			   boundingBox[2], boundingBox[3]);
			cv::Mat mat = image_to_mat(objCroppedImage);
			images.emplace_back(mat);
		}
    }

    return images;
}

void display(const std::vector<std::shared_ptr<op::Datum>>& datumsPtr,
			 cv::Mat& image)
{
	const std::map<unsigned int, std::string>& numBodyPartLabels = op::getPoseBodyPartMapping(POSE_MODEL);
    try {
    	const op::Array<float> & poseKeyPoints = datumsPtr.at(0)->poseKeypoints;
    	const int numPeople = datumsPtr.at(0)->poseKeypoints.getSize(0);
    	const int numPoints = datumsPtr.at(0)->poseKeypoints.getSize(1);

    	cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", datumsPtr.at(0)->cvOutputData);
    	cv::waitKey(10000);

		for (int p = 0; p < numPeople; ++p) {
			for (int n = 0; n < numPoints; ++n) {
				const double x = poseKeyPoints[{p, n, 0}];
				const double y = poseKeyPoints[{p, n, 1}];

				if( poseKeyPoints[{p, n, 2}] > 0.1) {
					cv::putText(image, numBodyPartLabels.at(n),  cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX,
								0.6, cv::Scalar(255,255,255));
				}
			}
		}

		// Display image
		cv::Point point;
		cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", image);
		cv::waitKey(20000);
    }
    catch (const std::exception& e) {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

//! returns image paths sorted by name
std::vector<std::string> getImagesInDir(const std::string& imageDir)
{
	std::vector<std::string> fileNames;
	namespace fs = std::experimental::filesystem;
	for (const auto & entry : fs::directory_iterator(imageDir)) {
		//! image sequence number
		//! TODO: get file extension (may be)
		const std::string filename = entry.path().filename();
		fileNames.emplace_back(filename);

	}

	const auto fileSort = [] (const std::string & file1, const std::string & file2) -> bool {
		return std::stoi(file1) < std::stoi(file2);
	};

	std::sort(fileNames.begin(), fileNames.end(), fileSort);
	return fileNames;
}

void writeBboxesToFile(const std::string& outFilePath,
					   const std::vector<std::array<double, BBOX_PARAMETERS>>& personBboxes,
					   const std::string& frameNumberPath)
{
	const int frameNumber = std::stoi(frameNumberPath);
	std::cout << frameNumberPath << ", " << frameNumber << ", \n";
	std::ofstream outFile;
	outFile.open(outFilePath, std::ios_base::app);
	//! write bboxes
	int bboxId = 1;
	for (const auto & box : personBboxes) {
		outFile << frameNumber << ", -1, " << box[0] << ", " << box[1] << ", "
				<< box[2] << ", " << box[3] << ", -1, -1, -1, -1"
				<< '\n';
		++bboxId;
	}

	outFile.close();

}

int main() {

	std::unique_ptr<network> net = configureNetwork();
	if(!net) {
		std::cout << "Network not initialised\n";
	}

	//! returns a file list in a directory sorted by numbers
	//! assumes filenames contains numbers
	const std::vector<std::string> imagePaths = getImagesInDir(VIDEO_PATH);
	for(const auto & entry : imagePaths) {
		int numBoxes = 0;
		image im = loadImage(VIDEO_PATH + entry);
		//! TODO - remove pointer
		detection* dets = detectObjectsInImage(numBoxes, net, im);
		//! TODO transform this into exception
		if(!dets) {
			std::cout << __LINE__ << ": detection is null \n";
		}
		//! yolo boiler plate code
		const int numClasses = net->layers[net->n-1].classes;
		do_nms_sort(dets, numBoxes, numClasses, THRESHOLD);
		std::vector<std::array<double, BBOX_PARAMETERS>> personBboxes =
											getPersonBboxes(im, dets, numBoxes);
		//! list of files
		writeBboxesToFile(DETECTIONS_FILE, personBboxes, entry);
	}





//	std::vector<cv::Mat>croppedCvImages = getImageLabels(im, dets, numBoxes);
	/*
	// openpose section
	{
		op::Wrapper opWrapper(op::ThreadManagerMode::Asynchronous);

		// blocking call. ensure openpose configuration is complete
		opWrapper.start();


		typedef std::vector<std::shared_ptr<op::Datum>>		Datums;
		typedef std::shared_ptr<Datums>						SpDatums;

		for (size_t i = 0; i < croppedCvImages.size(); ++i) {
			SpDatums datumProcessed = opWrapper.emplaceAndPop(croppedCvImages[i]);
			if (datumProcessed != nullptr && !datumProcessed->empty()) {
				display(*datumProcessed.get(), croppedCvImages[i]);
			}
		}
	}
	*/

}
