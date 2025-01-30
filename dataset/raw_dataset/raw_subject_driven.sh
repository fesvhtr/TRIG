mkdir subject-driven/Subjects200K
~/hfd.sh Yuanshi/Subjects200K --dataset --local-dir subject-driven/Subjects200K -x 16
~/hfd.sh Yuanshi/Subjects200K_collection3 --dataset --local-dir subject-driven/Subjects200K_collection3 -x 16

mkdir subject-driven/X2I-subject-driven

~/hfd.sh yzwang/X2I-subject-driven --dataset --local-dir subject-driven/X2I-subject-driven --include retrieval* -x 16
cd subject-driven/X2I-subject-driven/retrieval
tar -zxvf download_images.tar.gz
tar -zxvf download_images_two.tar.gz
rm -rf download_images.tar.gz download_images_two.tar.gz
cd ../../..

~/hfd.sh yzwang/X2I-subject-driven --dataset --local-dir subject-driven/X2I-subject-driven --include character* -x 16
cd subject-driven/X2I-subject-driven/character
tar -zxvf character.tar.gz
rm -rf character.tar.gz
cd ../../..

cd subject-driven
git clone https://github.com/google/dreambooth.git