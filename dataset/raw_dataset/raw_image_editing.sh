mkdir image-editing/X2I-mm-instruction

~/hfd.sh yzwang/X2I-mm-instruction --dataset --local-dir image-editing/X2I-mm-instruction --include magicbrush -x 16
cd image-editing/X2I-mm-image-editing/magicbrush
tar -zxvf magicbrush.tar.gz
rm -rf magicbrush.tar.gz
cd ../../..

~/hfd.sh yzwang/X2I-mm-instruction --dataset --local-dir image-editing/X2I-mm-instruction --include pix2pix -x 16
cd image-editing/X2I-mm-image-editing/pix2pix
cat images.tar.gz.* > images.tar.gz
tar -zxvf images.tar.gz
rm -rf images.tar.gz images.tar.gz.*
cd ../../..

~/hfd.sh yzwang/X2I-mm-instruction --dataset --local-dir image-editing/X2I-mm-instruction --include stylebooth* -x 16
cd image-editing/X2I-mm-image-editing/stylebooth
tar -zxvf stylebooth.tar.gz
rm -rf stylebooth.tar.gz
cd ../../..

mkdir image-editing/OmniEdit-Filtered
~/hfd.sh TIGER-Lab/OmniEdit-Filtered-1.2M --dataset --local-dir image-editing/OmniEdit-Filtered --include data/dev-00000-of-00001.parquet -x 16

