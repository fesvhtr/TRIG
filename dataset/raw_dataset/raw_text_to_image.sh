mkdir text-to-image/DOCCI
wget -c -P text-to-image/DOCCI https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines
wget -c -P text-to-image/DOCCI https://storage.googleapis.com/docci/data/docci_images.tar.gz
wget -c -P text-to-image/DOCCI https://storage.googleapis.com/docci/data/docci_metadata.jsonlines

mkdir text-to-image/X2I-text-to-image
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include .gitattributes README.md 
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include laion-coco-aesthetic.jsonl
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include laion-coco-aesthetic/0000* 
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include laion-coco-aesthetic/0001* 
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include laion-coco-aesthetic/0002* 
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include laion-coco-aesthetic/0003* 
~/hfd.sh yzwang/X2I-text-to-image --dataset --local-dir text-to-image/X2I-text-to-image --include laion-coco-aesthetic/0004* 

cd text-to-image/X2I-text-to-image/laion-coco-aesthetic
for i in $(seq -f "%05g" 0 49); do
    tar -zxvf ${i}.tar.gz;
done
rm -rf *.tar.gz

cd text-to-image/X2I-text-to-image/laion-coco-aesthetic
for i in $(seq -f "%05g" 20 49); do
    tar -zxvf ${i}.tar.gz;
done
