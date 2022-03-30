import os
import tensorflow as tf
import socket
import select
import numpy
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

model = tf.keras.models.load_model("model/")

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

listener.bind((socket.gethostbyname(socket.gethostname()), 1024))

listener.listen()

print("listening...")

sockets = [listener]

while True:

    ready = select.select(sockets, [], [], 10000000)[0]

    if ready:

        try:

            listener.settimeout(0.1)

            new_sock = (listener.accept()[0])

            sockets.append(new_sock)

            listener.settimeout(None)

            print("client has connected: " + str(new_sock))

        except:

            listener.settimeout(None)

        sockets_count = len(sockets)

        if sockets_count > 1:

            for cur in range(1, sockets_count):

                try:

                    sockets[cur].send('0'.encode())

                    img_bytes = sockets[cur].recv(5000000)
        
                except Exception:

                    print("client has disconnected: " + str(sockets[cur]))

                    sockets.remove(sockets[cur])

                    continue

                if not img_bytes:

                    continue

                img_arr = numpy.frombuffer(img_bytes, numpy.uint8)

                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
                hue = (img_hsv[:, :, 0] < 100) * (img_hsv[:, :, 0] > 3)
            
                saturation = (img_hsv[:, :, 1] > 70)
            
                value = (img_hsv[:, :, 2] > 120)
            
                mask = saturation * value * hue
            
                img_hsv[:, :, 0] *= mask
            
                img_hsv[:, :, 1] *= mask
            
                img_hsv[:, :, 2] *= mask
            
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

                img = cv2.resize(img, (224, 224))

                img = numpy.expand_dims(img, axis = 0)

                print(model(img))