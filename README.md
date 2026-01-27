# EGA-Net: Edge-Guided Attention with Bi-Axial Strip Pooling and Context Pyramid for Retinal Vessel Segmentation

<img width="4761" height="2494" alt="image1" src="https://github.com/user-attachments/assets/216fc5ab-b4a4-41b1-9831-951538135ada" />

Installation

torch==2.6.0+cu124

numpy==2.1.2

opencv-python==4.11.0

matplotlib==3.10.0

Pillow==11.1.0

scikit-learn==1.6.1

pandas==2.3.2

tensorboard==2.20.0

database

https://github.com/wangjinweige123/EGA-Net/releases/download/database-v1.0/Chase.zip

https://github.com/wangjinweige123/EGA-Net/releases/download/database-v1.1/Drive.zip

https://github.com/wangjinweige123/EGA-Net/releases/download/database-v1.2/Stare.zip

Training and Evaluation

python eval_chase.py --use_saspp --weight_path "Chase/test/checkpoint/UNet_saspp.pth" 

python eval_chase.py --use_ega --weight_path "Chase/test/checkpoint/UNet_ega.pth"  

python eval_chase.py --use_saspp --use_ega --weight_path "Chase/test/checkpoint/UNet_saspp_ega.pth" 


