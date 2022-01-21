from azure.cognitiveservices.vision.face import FaceClient
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from msrest.authentication import CognitiveServicesCredentials


API_KEY = ""  #enter your azure cloud service subscription key here
ENDPOINT = ""  #enter your azure cloud service subscription endpoint here
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))


def show_faces(image_path, detected_faces, show_id=False):
   
    # Open an image(from main program)
    img = Image.open(image_path)

    # Create a figure to display the results
    fig = plt.figure(figsize=(8, 6))

    if detected_faces:
        # If there are faces, how many?
        num_faces = len(detected_faces)
        prediction = ' (' + str(num_faces) + ' faces detected)'
        print(prediction)


        # Draw a rectangle around each detected face
        for face in detected_faces:
            r = face.face_rectangle
            bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
            draw = ImageDraw.Draw(img)
            draw.rectangle(bounding_box, outline='blue', width=5)
            if show_id:
                plt.annotate(face.face_id,(r.left, r.top + r.height + 15), backgroundcolor='white')
        #a = fig.add_subplot(1,1,1)
        fig.suptitle(prediction)

    plt.axis('off')
    plt.imshow(img)
    img.show()

#open an image
image = input("Enter the image location:")
image_path = open(image, 'rb')

# Detect faces
detected_faces = face_client.face.detect_with_stream(
    image = image_path,
    detection_model='detection_01',
    recognition_model='recognition_04'
    )

show_faces(image_path, detected_faces)


