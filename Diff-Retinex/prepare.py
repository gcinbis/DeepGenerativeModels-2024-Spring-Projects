from PIL import Image
import os
from pathlib import Path

def prepare(sizem,train):
    size = sizem
    def resize_image(input_path, output_path, size):
        with Image.open(input_path) as img:
            img_resized = img.resize(size)
            img_resized.save(output_path)
    def create_subdirectories(parent_dir, subdirs):
        try:
            # Ana klasörü oluştur
            parent_path = Path(parent_dir)
            parent_path.mkdir(parents=True, exist_ok=True)
            # Alt klasörleri oluştur
            for subdir in subdirs:
                path = parent_path / subdir
                path.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            print(f'{e}')

    if train =="eval":
        parent_directory = 'eval'
        subdirectories = ['highIllumination', 'lowIllumination', 'highReflectance', 'lowReflectance']
        create_subdirectories(parent_directory, subdirectories)



        input_folder = "./lol_dataset_prosessed_eval/highi"
        output_folder = "./eval/highIllumination"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")

        input_folder = "./lol_dataset_prosessed_eval/lowi"
        output_folder = "./eval/lowIllumination"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")

        input_folder = "./lol_dataset_prosessed_eval/highr"
        output_folder = "./eval/highReflectance"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")

        input_folder = "./lol_dataset_prosessed_eval/lowr"
        output_folder = "./eval/lowReflectance"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")


    else:
        parent_directory = 'dataset'
        subdirectories = ['highIllumination', 'lowIllumination', 'highReflectance', 'lowReflectance']

        create_subdirectories(parent_directory, subdirectories)

        input_folder = "./lol_dataset_prosessed/highi"
        output_folder = "./dataset/highIllumination"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")

        input_folder = "./lol_dataset_prosessed/lowi"
        output_folder = "./dataset/lowIllumination"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")

        input_folder = "./lol_dataset_prosessed/highr"
        output_folder = "./dataset/highReflectance"


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")

        input_folder = "./lol_dataset_prosessed/lowr"
        output_folder = "./dataset/lowReflectance"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            if os.path.isfile(input_path):
                try:
                    resize_image(input_path, output_path, size)
                except Exception as e:
                    print(f"{filename} Error: {e}")


