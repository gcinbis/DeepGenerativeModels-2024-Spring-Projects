#!/bin/bash

# İndirilecek dosyanın URL'si
file_url="https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip?download=true"

# İndirilecek dosyanın adını belirleyin
output_file="lol_dataset.zip"

# İndirme işlemi
wget "$file_url" -O "$output_file"

wget --no-check-certificate "https://drive.google.com/file/d/1nOLIiVGoioCYgXrwlPduWcqGglzo3E0d/view?usp=sharing"

wget --no-check-certificate "https://drive.google.com/file/d/1Mftlzs3MHkFOUTQjXYDe2iF35eJnj9gQ/view?usp=sharing"

wget --no-check-certificate "https://drive.google.com/file/d/1EMwq_eHhZy2xmjUHhRCDnhfQDSvWcz99/view?usp=sharing"

# İndirilen dosyayı çıkartma
unzip "$output_file" -d ./

# İndirilen zip dosyasını silebilirsiniz (isteğe bağlı)
rm "$output_file"

