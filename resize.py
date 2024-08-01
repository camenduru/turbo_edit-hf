from PIL import Image

def resize_image(input_path, output_path, new_size):
    # Open the image
    image = Image.open(input_path)

    # Resize the image
    resized_image = image.resize(new_size)

    # Save the resized image
    resized_image.save(output_path)

# Example usage
input_path = "demo_im/pexels-pixabay-458976.png"
output_path = "demo_im/pexels-pixabay-458976.less.png"
new_size = (512, 512)

resize_image(input_path, output_path, new_size)