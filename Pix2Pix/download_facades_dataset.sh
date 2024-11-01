FILE=night2day  # 数据集名称
TAR_FILE=/home/SA24001013/datasets/$FILE.tar.gz  # 数据集文件的路径
TARGET_DIR=/home/SA24001013/datasets/$FILE/  # 解压目标目录

# 创建目标目录
mkdir -p $TARGET_DIR

# 下载数据集
echo "Downloading dataset..."
curl -L http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz -o $TAR_FILE

# 解压数据集
tar -zxvf $TAR_FILE -C $TARGET_DIR

# 删除压缩包
rm $TAR_FILE

# 生成文件列表
find "${TARGET_DIR}train" -type f -name "*.jpg" | sort -V > /home/SA24001013/train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" | sort -V > /home/SA24001013/val_list.txt