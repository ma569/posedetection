/*
 * detct_objDetect.h
 *
 *  Created on: Aug 2, 2019
 *      Author: musabbir
 */

#ifndef DETCT_OBJDETECT_H_
#define DETCT_OBJDETECT_H_

//#include "darknet.h"

namespace detct {

class ObjDetect {
public:
	ObjDetect();
	virtual ~ObjDetect();

	ObjDetect(const ObjDetect&) = delete;

	ObjDetect& operator=(const ObjDetect&) = delete;

	ObjDetect(const ObjDetect&&) = delete;
};

} /* namespace detct */

#endif /* DETCT_OBJDETECT_H_ */
