import cv2
import numpy as np
import dlib

modelPath = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)
def getFaces(faceDetector, landmarkPredictor, im, DOWNSAMPLE_RATIO = 1):
    faces = []
    facespoints = []
    facesoffsets = []
    facespointsratio = []
    gaze = []
    (realheight,realwidth) = im.shape[:2]
    imSmall = im
    padfactor = 0.2
    if (DOWNSAMPLE_RATIO != 1):
        imSmall = cv2.resize(im,None,fx=1.0/DOWNSAMPLE_RATIO,fy=1.0/DOWNSAMPLE_RATIO, interpolation = cv2.INTER_LINEAR)
    faceRects = faceDetector(imSmall, 0)
    for i,face in enumerate(faceRects):
        faceRect = [face.left(),face.top(),face.right(),face.bottom()]
        height = faceRect[3] - faceRect[1];
        width = faceRect[2] - faceRect[0];
        (addy,addx) = (int(height*padfactor),int(width*padfactor))       #increase the cropping area of the face
        (fl,ft,fr,fb) = ((max(0,(faceRect[0]-addx)*DOWNSAMPLE_RATIO)),
                         (max(0,(faceRect[1]-addy)*DOWNSAMPLE_RATIO)),
                         (min(realwidth,(faceRect[2]+addx)*DOWNSAMPLE_RATIO)),
                         (min(realheight,(faceRect[3]+addy)*DOWNSAMPLE_RATIO)))
        croppedface = im[ft:fb,fl:fr].copy()
        ratio = 300.0/croppedface.shape[1]
        croppedface = cv2.resize(croppedface,None,fx=ratio,fy=ratio, interpolation = cv2.INTER_LINEAR)
        faces.append(croppedface)
        (landmarks,eyepos) = getLandmarks(faceDetector,landmarkPredictor,croppedface)
        facespoints.append(landmarks)
        facesoffsets.append([fl,ft])
        facespointsratio.append(1/ratio)
        gaze.append(eyepos)
    return (faces,facespoints,facesoffsets,facespointsratio,gaze)
def getLandmarks(faceDetector, landmarkPredictor, im):
    landmarksarr = []
    faceRects = faceDetector(im, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cutofffactor = 10 #Workaround to Remove false errors
    eyepos = []
    for i,face in enumerate(faceRects): 
        landmarks = landmarkPredictor(im, face)
        for p in landmarks.parts():
            pt = (int(p.x), int(p.y))
            landmarksarr.append(pt)
        for j,i in enumerate(landmarksarr):
            cv2.circle(im,(i[0],i[1]),1,(0,255,0),-1)  
            
        #ESTIMATE GAZE POSITION - Naive Method
        closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, iterations=3)
        lefteye = landmarksarr[36:42]
        righteye = landmarksarr[42:48]
        compensate = 0
        
        (x, y, w, h) = cv2.boundingRect(np.array(lefteye))
        cutoff = int(w/cutofffactor)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0))
        lefteyeimg = closing[y-compensate:y+h+compensate,x+cutoff:x+w-cutoff]
        eyepos.append(int(lefteyeimg.sum(axis=0)[2:].argsort()[:5].mean()))
        cv2.rectangle(im,(x+cutoff+eyepos[0]-1,y-compensate),(x+cutoff+eyepos[0]+1,y+compensate+h),255,-1)
        eyepos[0] = eyepos[0]*1.0/(w-cutoff-cutoff)

        (x, y, w, h) = cv2.boundingRect(np.array(righteye))
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255))
        righteyeimg = closing[y-compensate:y+h+compensate,x+cutoff:x+w-cutoff]
        eyepos.append(int(righteyeimg.sum(axis=0)[2:].argsort()[:5].mean()))
        cv2.rectangle(im,(x+cutoff+eyepos[1]-1,y-compensate),(x+cutoff+eyepos[1]+1,y+compensate+h),255,-1)
        eyepos[1] = eyepos[1]*1.0/(w-cutoff-cutoff)
        
        break #just need 1 face for this function
    return landmarksarr, eyepos
def drawEyes(image,facespoints,facesoffsets,facespointsratio,gaze,filtereyes=1):
    eyeavgpos = -1
    (ih,iw) = image.shape[:2]
    for i,face in enumerate(facespoints):
        if (len(gaze[i]) == 0):
            return -1 #no gaze detected
        for j,record in enumerate(face):
            face[j] = (int(facesoffsets[i][0]+int(facespointsratio[i]*record[0])),
                       int(facesoffsets[i][1]+int(facespointsratio[i]*record[1])))
        lefteye = face[36:42]
        righteye = face[42:48]
        (x, y, w, h) = cv2.boundingRect(np.array(lefteye))
        cv2.rectangle(image,(x,y+h),(x+w,y+h+2),(0,255,0))
        eyepos = int(gaze[i][0]*w)
        cv2.rectangle(image,(x+eyepos-1,y+1),(x+eyepos+1,y+h-1),(0,0,255),-1)
        
        (x, y, w, h) = cv2.boundingRect(np.array(righteye))
        cv2.rectangle(image,(x,y+h),(x+w,y+h+2),(0,255,0))
        eyepos = int(gaze[i][1]*w)
        cv2.rectangle(image,(x+eyepos-1,y+1),(x+eyepos+1,y+h-1),(0,0,255),-1)
        
        eyeavgpos = round(np.mean(gaze[i]),2)
        cv2.putText(image, str(eyeavgpos), (int(iw/2)-5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        if (i >= filtereyes-1):
            break
    
    return eyeavgpos

threshleft = -1
threshright = -1
gazeavg = []
gazemin = -1
gazemax = -1
startstate = -1
calibrationstate = 0

cv2.namedWindow('Speaking Eyes', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Speaking Eyes',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
questionlist = [("Are you ready?",1),
                ("Is 1 + 1 = 11?", 0),
                ("Is 2 + 2 = 4?", 1)
               ]
currentqn = 0
freezescreen = ""
ro = 80 #camera offset
camera = cv2.VideoCapture(0)
while True:
    (grabbed, frame) = camera.read()
    if (grabbed==False):
        break
    ratio = 720/frame.shape[1]
    img = cv2.resize(frame,None,fx=ratio,fy=ratio, interpolation = cv2.INTER_LINEAR)
    img = cv2.flip(img,flipCode=1)
    (h,w) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    (croppedfaces,facespoints,facesoffsets,facespointsratio,gaze) = getFaces(detector, predictor, gray, 2)
    gazevalue = drawEyes(img, facespoints,facesoffsets,facespointsratio,gaze)
    adjustingval = abs(gazevalue - np.mean(gazeavg))
    if (startstate == -1):
        startstate = gazevalue
    if (gazevalue>0):
        gazeavg.insert(0,gazevalue)
    gazeavg = gazeavg[:20]
    gazevalue = round(np.mean(gazeavg),2)
    if (np.isnan(gazevalue)):
        gazevalue = -1
    gazeinstant = np.mean(gazeavg[:3])
    if (np.isnan(gazeinstant)):
        gazeinstant = -1

    calibrated = False
    if gazevalue < gazemin or gazemin==-1:
        gazemin = gazevalue
    if gazevalue > gazemax or gazemax==-1:
        gazemax = gazevalue
    threshleft = round(gazemin+((gazemax-gazemin)*1.0/3),2)
    threshright = round(gazemin+((gazemax-gazemin)*2.0/3),2)
    if (len(gazeavg)>=20):
        if (calibrationstate==2 and threshright-threshleft>=0.025):
            calibrated = True
            #YES region
            cv2.rectangle(img,(20,200),(120,300),(0,255,0),-1)
            cv2.rectangle(img,(22,202),(118,298),(255,255,255),-1)
            cv2.putText(img, "Yes", (40,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            #NO region
            cv2.rectangle(img,(610-ro,200),(710-ro,300),(0,255,0),-1)
            cv2.rectangle(img,(612-ro,202),(708-ro,298),(255,255,255),-1)
            cv2.putText(img, "No", (625-ro,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
            #CENTER region
            cv2.rectangle(img,(int(w/2)-100,50),(int(w/2)+100,120),(255,255,255),-1)
            cv2.putText(img, questionlist[currentqn][0], (int(w/2)-90,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(img, "Select Yes or No", (int(w/2)-90,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                   
            if (freezescreen != ""):
                cv2.rectangle(img,(int(w/2)-100,50),(int(w/2)+100,120),(0,255,0),-1)
                cv2.putText(img, questionlist[currentqn][0], (int(w/2)-90,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                cv2.putText(img, "Answer Received", (int(w/2)-90,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                # TODO: Encouragement when answer is correct/received
            else:
                if (gazeinstant>threshright):
                    cv2.rectangle(img,(612-ro,202),(708-ro,298),(0,255,0),-1)
                    cv2.putText(img, "No", (632-ro,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    # TODO: Check whether answer is correct/received
                elif (gazeinstant<threshleft): 
                    cv2.rectangle(img,(22,202),(118,298),(0,255,0),-1)
                    cv2.putText(img, "Yes", (38,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    # TODO: Check whether answer is correct/received
                else:
                    cv2.rectangle(img,(int(w/2)-100,50),(int(w/2)+100,120),(0,255,0),-1)
                    cv2.putText(img, questionlist[currentqn][0], (int(w/2)-90,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    cv2.putText(img, "Select Yes or No", (int(w/2)-90,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        else:
            calibrated = False

    if (calibrated == False):
        if (calibrationstate == 0):    #left not yet calibrated
            if ((gazeinstant<threshleft-0.01 and adjustingval<0.01 and len(gazeavg)>=10)):
                calibrationstate = 1   #right
            #Ask user to calibrate        
            cv2.rectangle(img,(10,200),(110,300),(0,255,0),-1)
            cv2.putText(img, "Look here", (15,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(img, "calibrating", (15,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(img, str(round(adjustingval,2)) + " " + str(gazemin), (15,280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
            
        elif (calibrationstate == 1):
            if ((gazeinstant>threshright+0.01 and adjustingval<0.01 and len(gazeavg)>=10)):   #right not yet calibrated  
                calibrationstate = 2
            cv2.rectangle(img,(612-ro,202),(708-ro,298),(0,255,0),-1)
            cv2.putText(img, "Look here", (618-ro,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(img, "calibrating", (618-ro,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(img, str(round(adjustingval,2)) +" " + str(gazemax), (618-ro,280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        elif (calibrationstate == 2):
            cv2.rectangle(img,(int(w/2)-100,h-90),(int(w/2)+100,h-20),(0,255,0),-1)
            cv2.putText(img, "Calibration failed?", (int(w/2)-90,h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
            cv2.putText(img, "range not wide enough", (int(w/2)-90,h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    cv2.circle(img, (int(w/2),int(h/2)), 2, (255,255,0), -1)
    cv2.putText(img, str(threshleft), (int(w/2)-20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(img, str(threshright), (int(w/2)+20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    
    cv2.imshow("Speaking Eyes",img)
    
    key = cv2.waitKey(20)
    if key&0xFF == ord("q"):
        break    
    elif key&0xFF == ord("r"):
        calibrationstate = 0
        gazemin = -1
        gazemax = -1
        threshleft=-1
        threshright=-1
        gazeavg = []
        startstate = -1
        
camera.release()
cv2.destroyAllWindows()
