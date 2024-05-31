echo "Creating the training dataset directory"
mkdir data data/hr data/lr

echo "Downloading the training dataset"

echo "Downloading DIV2K dataset..."
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

echo "Unzipping DIV2K dataset..."
unzip DIV2K_train_HR.zip 
rm DIV2K_train_HR.zip
mv DIV2K_train_HR/* data/hr
rm -fr DIV2K_train_HR

echo "Downloading Flickr2K dataset..."
wget https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

echo "Unzipping Flickr2K dataset..."
tar -xvf Flickr2K.tar
rm Flickr2K.tar
mv Flickr2K/* data/hr
rm -fr Flickr2K

echo "Generating low res images"
python resize_4.py --folder data/hr --save_path data/lr

echo "Preprocessing the dataset"
python prepare_dataset.py --hr_folder data/hr --lr_folder data/lr --crop_size_hr 400 --crop_size_lr 100 --output_folder data/dataset_cropped

echo "Downloading the evaluation data that consists of Manga109, Set5, Set14, urban100, and DIV2K validation datasets"
mkdir data/evaluation 
mkdir data/evaluation/hr 
mkdir data/evaluation/lr 
mkdir data/evaluation/hr/manga109 
mkdir data/evaluation/lr/manga109 
mkdir data/evaluation/hr/Set5 
mkdir data/evaluation/lr/Set5 
mkdir data/evaluation/hr/Set14 
mkdir data/evaluation/lr/Set14 
mkdir data/evaluation/hr/urban100 
mkdir data/evaluation/lr/urban100 
mkdir data/evaluation/lr/DIV2K_valid_HR 
mkdir data/evaluation/lr/DIV2K_valid_HR 

echo "Downloading Manga109 dataset..."
gdown --id 1jw7QNbzea9SUq4IoRCO2q44TIyEXOl-f
unzip MANGA109.zip
rm MANGA109.zip
mv MANGA109/* data/evaluation/hr/manga109
python resize_4.py --folder data/evaluation/hr/manga109 --save_path data/evaluation/lr/manga109

echo "Downloading Set5, Set14, urban100, DIV2K validation"
gdown 1M-vP-rPNC0_DOZRf3ZQMzNz1ge7eoJsD
unzip dataset.zip
rm dataset.zip
mv dataset/Set5/* data/evaluation/hr/Set5
mv dataset/Set14/* data/evaluation/hr/Set14
mv dataset/urban100/* data/evaluation/hr/urban100
mv dataset/DIV2K_valid_HR/* data/evaluation/hr/DIV2K_valid_HR

python resize_4.py --folder data/evaluation/hr/Set5 --save_path data/evaluation/lr/Set5
python resize_4.py --folder data/evaluation/hr/Set14 --save_path data/evaluation/lr/Set14
python resize_4.py --folder data/evaluation/hr/urban100 --save_path data/evaluation/lr/urban100
python resize_4.py --folder data/evaluation/hr/DIV2K_valid_HR --save_path data/evaluation/lr/DIV2K_valid_HR

rm -fr dataset

echo "Downloading VGG network"
mkdir pretrained_models
gdown --id 1henrktM4Cw9hJIJBDEObAzl-eCbpzNaJ 
mv vgg16.pth pretrained_models/vgg16.pth

echo "Downloading the pre-trained model"
gdown --id 1o5qMZ3B-Z-VfTOj0rqxMqk4tL8o3oo6z
unzip model_weights.zip
rm model_weights.zip
echo "All done!"