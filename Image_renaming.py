import os
import csv

# # Function to rename the images
def rename_images():
    path = "Screenshots/test/augmented_not_human"
    files = os.listdir(path)
    i = 5023
    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, "Screenshot_" + str(i) + ".png"))
        i += 1

# Function to write the new names to a csv file
def write_to_csv():
    path = "Screenshots/test/augmented_not_human"
    files = os.listdir(path)
    with open("Image_classification_2.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["New Name", "Image Classification"])
        for file in files:
            writer.writerow([file, ""])

# Main function
def main():
    rename_images()
    write_to_csv()

if __name__ == "__main__":
    main()