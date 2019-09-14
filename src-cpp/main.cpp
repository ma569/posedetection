#include <memory>
#include <vector>
#include <sstream>
#include <iostream>

//#include <opencv2/opencv.hpp>

#include <openpose/headers.hpp>
#include <darknet.h>

#include "detct_objDetect.h"

const int MAX_CHAR_LEN             = 512;
const std::string CFG_FILE         = "/home/musabbir/workspace/poseDetection/darknet/cfg/yolov3.cfg";
const std::string WEIGHT_FILE      = "/home/musabbir/workspace/poseDetection/darknet/yolov3.weights";
const std::string IMAGE_FILE       = "/home/musabbir/workspace/poseDetection/images/crawl0.jpg";
const std::string CLASS_NAMES_FILE = "/home/musabbir/workspace/poseDetection/darknet/data/coco.names";
const op::PoseModel POSE_MODEL     = op::PoseModel::BODY_25;


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

std::vector<cv::Mat> getImageLabels(image im, detection *dets, int numBoxes, float thresh)
{
	char classNamesFileTemp[MAX_CHAR_LEN];
	strcpy(classNamesFileTemp, CLASS_NAMES_FILE.c_str());
	std::vector<cv::Mat> images;

    for(int i = 0; i < numBoxes; ++i) {
		if (dets[i].prob[0] > thresh) { //! prob[0] is 'person'
			//! guaranteed to be a person
			box objBoundingBox = dets[i].bbox;

			//! add max/min logic
            const int width   = objBoundingBox.w * im.w;
            const int height  = objBoundingBox.h * im.h;
            const int dx      = (objBoundingBox.x * im.w - 0.5 * width) < 0. ? 0.
            		 	 	 	 : (objBoundingBox.x * im.w - 0.5 * width);
            const int dy      = (objBoundingBox.y * im.h - 0.5 * height) < 0. ? 0.
            					 : (objBoundingBox.y * im.h - 0.5 * height);

			image objCroppedImage = crop_image(im, dx, dy, width, height);
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

//		for (int p = 0; p < numPeople; ++p) {
//			for (int n = 0; n < numPoints; ++n) {
//				const double x = poseKeyPoints[{p, n, 0}];
//				const double y = poseKeyPoints[{p, n, 1}];
//
//				if( poseKeyPoints[{p, n, 2}] > 0.1) {
//					cv::putText(image, numBodyPartLabels.at(n),  cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX,
//								0.6, cv::Scalar(255,255,255));
//				}
//			}
//		}

		// Display image
//		cv::Point point;
//		cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", image);
//		cv::waitKey(20000);
    }
    catch (const std::exception& e) {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int main() {

	detct::ObjDetect obj;

	char cfgFileTemp[MAX_CHAR_LEN];
	char weightFileTemp[MAX_CHAR_LEN];
	std::unique_ptr<network> net(load_network(strcpy(cfgFileTemp, CFG_FILE.c_str()),
											  strcpy(weightFileTemp, WEIGHT_FILE.c_str()),
											  0));
	if(!net) {
		std::cout << "Network not initialised\n";
	}


	char imageFileTemp[MAX_CHAR_LEN];
	strcpy(imageFileTemp, IMAGE_FILE.c_str());
	image im = load_image_color(imageFileTemp, 0, 0);
	image sized = letterbox_image(im, net->w, net->h);
	float* netDataNum = sized.data;
	if(!netDataNum) {
		std::cout << "netDataNum is null \n";
	}

	set_batch_network(net.get(), 1); //! 1 = batch size
	network_predict(net.get(), netDataNum);

	float threshold = 0.5;
	float hierThreshold = 0.5;
	int numBoxes = 0;
	detection *dets = get_network_boxes(net.get(), im.w, im.h, threshold,
									    hierThreshold, 0, 1, &numBoxes);
	if(!dets) {
		std::cout << __LINE__ << ": detection is null \n";
	}

	const int numClasses = net->layers[net->n-1].classes;
	do_nms_sort(dets, numBoxes, numClasses, threshold);
	std::vector<cv::Mat>croppedCvImages = getImageLabels(im, dets, numBoxes,
														 threshold);

	// openpose section
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
