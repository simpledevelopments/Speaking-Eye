# Speaking-Eye v0.1
Enable eye-controlled communications without Infrared.

With the advance of machine vision and deep learning, it should be feasible to track eye movements even without an infrared camera. Developing a cost effective solution that does not require the use of infrared exposure would be something helpful.

The code here is an example of a naive solution, where the eyes can be used to control a computer to make a selection out of 3 options: "Yes / No / What is your question?"
Code shared here is to promote research. For commercial projects, please contact us privately.


# How the demo looks like
(Currently runs between 8 to 15 frames per second (FPS) on a laptop with a webcam.)  
  
![speakingeye](https://user-images.githubusercontent.com/4750005/31617486-eda1b864-b2c1-11e7-9256-4b88df01f273.jpg)

# System Requirements
Install OpenCV and dlib with python bindings.  
If you encounter difficulties installing OpenCV or dlib, you can refer to these helpful guides:
- https://www.learnopencv.com/tag/dlib/
- https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

Place the trained dlib model 'shape_predictor_68_face_landmarks.dat' inside the subfolder 'models'

To run, python speakingeye.py


# Special Thanks

We are thankful for the useful learning resources provided by PyImageSearch and LearnOpenCV.  
Special Thanks to Adrian (PyImageSearch) and Satya (LearnOpenCV) for their excellent educational resources and blogs.


# Future Work

- A recommendation for future work would be to explore using a predictive Convolutional Neural Networks (CNN) on the eye to localize the eye center more precisely and accurately. Much better results would be expected given the advancement of CNNs in recent years.
- Hopefully, this will allow for much more precise controls, such as mouse control that would open up the possibilities beyond 3 simple options (yes/no/maybe). 
- Another possible improvement would be to allow the user to trigger resets or other special commands using eye gestures (e.g. blinking twice etc.)
- Tracking may be another approach to improve stabilization.
