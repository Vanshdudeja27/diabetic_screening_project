import base64

# Change this path to your image file location on your computer
image_path = r"C:\Users\vansh\Downloads\e4e343eaae2a.jpg"


with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    print(encoded_string)
    print("hello world")
