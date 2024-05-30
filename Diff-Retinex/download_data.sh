#!/bin/bash

# İndirilecek dosyanın URL'si
file_url="https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip?download=true"

# İndirilecek dosyanın adını belirleyin
output_file="lol_dataset.zip"

# İndirme işlemi
wget "$file_url" -O "$output_file"

# İndirilen dosyayı çıkartma
unzip "$output_file" -d ./

# İndirilen zip dosyasını silebilirsiniz (isteğe bağlı)
rm "$output_file"

