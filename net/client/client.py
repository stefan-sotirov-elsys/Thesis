import socket
import cv2
import numpy

def program_exit():

    connection.close()

    print("exiting...")

    quit()

connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connection.bind((socket.gethostbyname(socket.gethostname()), 0))

try:

    connection.connect((socket.gethostbyname(socket.gethostname()), 1024))

except:

    print("connection could not be established")

    program_exit()

print("connection has been established")

while True:

    path = input("path to file: ")

    if path == '':

        program_exit()

    try:

        img = cv2.imread(path)

        success, img_bytes = cv2.imencode(".jpg", img)

    except:

        print("image could not be loaded")

        continue

    if success:

        try:

            connection.sendall(img_bytes)

        except:
            
            print("connection to the server is lost")

            program_exit()

        print("sent " + path)

    else:
        
        print("image could not be loaded")
