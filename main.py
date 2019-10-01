import numpy as np
import os
import cv2


def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    print("get_path_list")
    listSubDirNames = []
    for filename in os.listdir(root_path):
        # print(filename)
        listSubDirNames.append(filename)

    return listSubDirNames
    



def get_class_names(root_path, train_names):
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''
    print("get_class_names")
    listAllImgPathTrain = []
    for f in train_names:
        for fi in os.listdir(root_path+'/'+f):
            listAllImgPathTrain.append(root_path+'/'+f+'/'+fi)
    listImageClassId = []
    for image_id,f in enumerate(train_names):
        for fi in os.listdir(root_path+'/'+f):
            # print(str(image_id) + ' <- ' + f )
            # image_id nantinya digunakan sebagai label jawaban saat training.
            listImageClassId.append(image_id)

    return listAllImgPathTrain,listImageClassId
    

def get_train_images_data(image_path_list):
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''
    print("get_train_images_data")
    list_img = []
    for i in image_path_list:
        img = cv2.imread(i)
        list_img.append(img)
    return list_img


def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returnsc
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    print("detect_faces_and_filter")
    listFilteredCroppedFaces = []
    listFilteredFacesLocations = []
    listFilteredImageIds = []
    for idx,im in enumerate(image_list):
        # convert to grayscale
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # os.getcwd (get current working directory, ambil path saat ini untuk akses file)
        faceCascade = cv2.CascadeClassifier(os.getcwd()+'/'+"haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(im, scaleFactor=1.2, minNeighbors=5)
        # filter allow only single face detected
        if(len(faces) == 1):
            for face_rectangle in faces:
                x,y,w,h = face_rectangle
                # cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,255))
                # crop face
                cropped_face = im[y:y+h, x:x+w]
            
            listFilteredCroppedFaces.append(cropped_face)
            listFilteredFacesLocations.append([x,y,w,h])
            if(image_classes_list!= None):
                listFilteredImageIds.append(image_classes_list[idx])
    return listFilteredCroppedFaces,listFilteredFacesLocations,listFilteredImageIds



def train(train_face_grays, image_classes_list):
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''
    print("train")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.train(train_face_grays,np.array(image_classes_list))

    return face_recognizer


def get_test_images_data(test_root_path, image_path_list):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''
    print("get_test_images_data")
    list_img = []
    for i in image_path_list:
        img = cv2.imread(test_root_path+'/'+ i)
        list_img.append(img)
    return list_img

def predict(classifier, test_faces_gray):
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    print("predict")
    predictionLabels = []
    for face in test_faces_gray:
        label,_ = classifier.predict(face)
        predictionLabels.append(label)
        
    return predictionLabels

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''
    print("draw_prediction_results")
    drawnImgs = []
    for res,img,[x,y,w,h] in zip(predict_results,test_image_list,test_faces_rects):
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        text = train_names[res]
        cv2.putText(img,text, (x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        drawnImgs.append(img)
    return drawnImgs


def combine_results(predicted_test_image_list):
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''
    print("combine_results")
    ct = 0
    for imgs in predicted_test_image_list:
        imgs = np.array(imgs)
        if(ct == 0):
            combined = imgs
        else:
            combined = np.hstack((combined,imgs))
        ct+=1

    return combined


def show_result(image):
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''
    print("show_result")
    cv2.imshow('Result',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    
    # train_root_path = "[PATH_TO_TRAIN_ROOT_DIRECTORY]"
    train_root_path = os.getcwd()+ "/dataset/dataset/train"

    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    # test_root_path = "[PATH_TO_TEST_ROOT_DIRECTORY]"
    test_root_path = os.getcwd() + "/dataset/dataset/test"
    

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)