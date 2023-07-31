import argparse
import cv2
import numpy as np 

# global variable 
# thresholds
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
class_names = []
model_in_w = 416
model_in_h = 416

# display parameters 
color_green = (0, 255, 0)
color_blue = (0, 0, 255)
font = cv2.FONT_ITALIC

def preprocess_frame(frame):
    blob_img = cv2.dnn.blobFromImage(frame, scalefactor = 1/255, size = (model_in_w, model_in_h), 
                                mean = (0, 0, 0), swapRB = True)
    return blob_img

def inference_on_frame(frame, net):
    boxes = []
    confidences = []
    class_id = []
    
    img_h, img_w = frame.shape[:2]

    # Preprocess 
    blob_img = preprocess_frame(frame)
    
    # Set model input 
    net.setInput(blob_img)
    
    # Do forward pass
    output_layer_names = net.getUnconnectedOutLayersNames()  # can be made global
    layer_outputs = net.forward(output_layer_names)
    
    # for each output
    for output in layer_outputs:
        # each layer has grid_size x grid_size x num_bb boxes
        for bbox_dtctn in output:

            cls_scores = bbox_dtctn[5:]
            max_id = np.argmax(cls_scores)
            confidence = cls_scores[max_id]

            if confidence > CONFIDENCE_THRESHOLD: 

                x_c = int(bbox_dtctn[0]*img_w) # Scale it up to fit in the image
                y_c = int(bbox_dtctn[1]*img_h)
                w = int(bbox_dtctn[2]*img_w)
                h = int(bbox_dtctn[3]*img_h)

                x_min = int(x_c - w/2)
                y_min = int(y_c - h/2)


                boxes.append([x_min, y_min, w, h]) # store the boxes
                confidences.append(float(confidence))
                class_id.append(max_id)
                
    
    # return indices of the bboxes which have appropriate threshold 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold = CONFIDENCE_THRESHOLD, 
                               nms_threshold = NMS_THRESHOLD)
    
    return (indexes, boxes, confidences, class_id)
    
def write_on_frame(frame, inference_output):
    
    indexes, boxes, confidences, class_id = inference_output
    
    if len(indexes)>0: 
        
        for i in indexes.flatten():

            # Retreive the parameter
            x_min, y_min, w, h = boxes[i]
            label = str(class_names[class_id[i]])
            conf = str(round(confidences[i],2))

            # Put Rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_min+w, y_min+h), color_green, thickness=1)

            # Put Text 
            cv2.putText(frame, label+ " "+conf, (x_min, y_min+10), font, 0.4, color_blue, thickness=1)


def video_camera_mode(*args):

    if mode == "CAMERA":
        # Open the webcam
        cap = cv2.VideoCapture(0)
    else: 
        video_pth = args[0]
        cap = cv2.VideoCapture(video_pth)

    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on frame
        infc_out = inference_on_frame(frame, model)

        # Write on frame
        write_on_frame(frame, infc_out)

        # Display the frame
        cv2.imshow("Output", frame)

        # Stop capturing frames when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def image_mode(img_pth): 

    frame = cv2.imread(img_pth)

    # Run inference on frame
    infc_out = inference_on_frame(frame, model)

    # Write on frame
    write_on_frame(frame, infc_out)

    # Display the frame
    cv2.imshow("Output", frame)

    # Stop capturing frames when 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
       cv2.destroyAllWindows()

def load_model(mdl_weight, cfg_file):
    # load model with opencv
    net = cv2.dnn.readNet(mdl_weight, cfg_file)
    return net



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert video to gif, can skip part of video as well")

    
    weight_pth ="./yolov3-tiny.weights"
    cfg_pth="./yolov3-tiny.cfg"
    labels_file = "./coco-labels_80.txt"

    parser.add_argument("-w", "--model_weight", help="darknet weight file location")
    parser.add_argument("-c", "--cfg_file", help="darknet model cfg file")
    parser.add_argument("-l", "--labels_file", help="class_labels list")
    parser.add_argument("-m", "--mode", help="mode for running the application")
    parser.add_argument("-i", "--img_pth", help="image file path")
    parser.add_argument("-v", "--video_pth", help="video file path")

    # Parse the command-line arguments
    args = parser.parse_args()

    weight_pth = args.model_weight
    if weight_pth is None:
        parser.error('Model weight path is missing.')

    cfg_pth = args.cfg_file
    if cfg_pth is None:
        parser.error('Model cfg path is missing.')

    labels_file = args.labels_file
    if weight_pth is None:
        parser.error('Model labels file is missing.')
    
    global mode
    mode = args.mode
    if mode is None: 
        mode = "CAMERA"

    img_pth = args.img_pth
    video_pth = args.video_pth


    with open(labels_file, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # load model 
    model = load_model(weight_pth, cfg_pth)

    if mode == "CAMERA":
        video_camera_mode()
    elif mode =="VIDEO": 
        if video_pth is None: 
            parser.error('Video file path is missing.')
        video_camera_mode(video_pth)
    elif mode == "IMAGE": 
        if img_pth is None: 
            parser.error('Image file path is missing.')
        image_mode(img_pth)
    else:
        print(mode, " is not an appropriate mode. The supported modes are 'IMAGE', 'CAMERA', VIDEO' .")


