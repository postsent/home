# ROS

![](pub_sub.png)

```
Terminal 1:    

    $ roscore

Terminal 2:

    $ python pub.py
Terminal 3:

    $ python sub.py
Terminal 4:

    $ python control.py

Then enter "m" or "s" for control.py
```

# code

```py3
#!/usr/bin/env python
# control.py
import rospy
from std_msgs.msg import String
from pub import movement_lock

def talker():
    
    pub = rospy.Publisher('lock', String)
    rospy.init_node('control')
    while not rospy.is_shutdown():
        global movement_lock
        a = raw_input("Move or stop enter m/s: ")
        if a == "m":
            pub.publish(String("1"))
            # movement_lock = False
        else:
            #movement_lock = True
            pub.publish(String("0"))

        rospy.sleep(1.0)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
```

```py3
# pub.py

import rospy
from std_msgs.msg import String

movement_lock = False

def callback(data):
    global movement_lock
    #rospy.loginfo("Received goal: %s",data.data[-1])
    if data.data == "0":
        movement_lock = True
    else:
        movement_lock = False

def talker():
    
    pub = rospy.Publisher('chatter', String)
    rospy.init_node('talker')
    rospy.Subscriber("lock", String, callback)

    while not rospy.is_shutdown():
        global movement_lock

        str = "Move to goal A"
        
        if not movement_lock:
            pub.publish(String(str))
            rospy.loginfo(str)
        else:
            str = "stop"
            pub.publish(str)
            rospy.loginfo(str)
        rospy.sleep(1.0)

if __name__ == '__main__':
    try:

        talker()
        
    except rospy.ROSInterruptException: pass
```

```py3
# sub.py
#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
def callback(data):
    rospy.loginfo("Received goal: %s",data.data[-1])

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("chatter", String, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
```


```py3
# laser scan

"""
Acknowledgement: This code example is by my lab tutor: Joshua Goncalves, from COMP3431
Link (dont recall if useful): https://www.youtube.com/watch?v=q3Dn5U3cSWk&feature=emb_logo&ab_channel=TheConstruct
"""
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

def callback(msg):

    # Min/Max Range of the LIDAR in metres
    # 120mm ~ 3500mm approximately 
    print("range_min: ", msg.range_min)
    print("range_max: ", msg.range_max)

    # Min/Max Angle of the LIDAR in radians
    # 0 radians to 6.28 (0 degrees to 360 degree)
    # Accuracy of 1 degree or 0.017 radians
    print('angle_min: ', msg.angle_min)
    print('angle_max: ', msg.angle_max)

    # The ranges have 360 values, one for each degree
    # The ranges are the distance in metres of the detected objects at each degree
    #       If an object is closer than 0.1199 metres or further away than 3.5 metres, it will appear as 0.0
    #       Distance accuracy also degrades the further away it is, but you probably dont need to worry too much about this
    print('len(ranges): ', len(msg.ranges))

    # The intensities have 360 values, one for each degree
    # The intensities are basically the reflectivity of the surface that is hit
    #       Surfaces that are highly reflective will reflect the majority of the beam and have a high intensity (like a mirror)
    #       Surfaces that do not have high reflectivity will have low intensity (like some black, matte material)
    # This might be useful in detecting walls based on their reflectivity, but I'm not too sure how sensitive the intensity is
    #       or how obvious a wall will be. Otherwise it can probably be used to just confirm that you've actually found an
    #       object based on its intensity.
    print('intensities: ', len(msg.intensities))

    target_distance = 0.1 # 0.1m or 100mm
    target_intensity = 4800.0 # You'll need to confirm this against the wall to ensure it's correct
    laser_ranges = msg.ranges

    # From the wheel end going in an anti-clockwise direction
    print('distance at angle 0: ',laser_ranges[0])
    print('distance at angle 45: ',laser_ranges[44])
    print('distance at angle 90: ',laser_ranges[90])
    # print('distance at angle 180: ', laser_ranges[180])
    # print('distance at angle 270: ',laser_ranges[270])
    for i in range(3):
        print()
rospy.init_node('laser_scan_example')
sub = rospy.Subscriber('/scan', LaserScan, callback)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
rospy.spin()
```

```py3
# face detection and stop robot
"Includes code from https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81"

import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

faceDetected = None

def face_detect(img_data):
    global faceDetected
    #print "callback"
    cvB = CvBridge()
    frame = cvB.imgmsg_to_cv2(img_data, "bgr8")

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #publish that faces were found
    if len(faces) > 0:
        person_pub.publish(Bool(True))
        faceDetected = True
        print("detected")
    else:
        if (faceDetected == True):
            person_pub.publish(Bool(False))
            print("Not detected")
            faceDetected = False
    

if __name__=='__main__':
    rospy.init_node("face_node")
    person_pub = rospy.Publisher("/movementLock", Bool, queue_size=1)

    sub_image = rospy.Subscriber("/raspicam_node/image", Image, face_detect, queue_size=1)

    rospy.spin()
```