import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import cv2

# 设置数据路径
train_dir = './data/train/'
val_dir = './data/validation/'

# 数据预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # 二分类：云海和日出
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 使用预训练模型VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 添加自定义层
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# 构建模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结预训练模型的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# 定义评分系统
def rate_scenery(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    print(f'Prediction: {prediction}')

    # 类别判断
    if prediction < 0.5:
        category = "云海"
    else:
        category = "日出"

    # 简单评分系统
    if prediction < 0.2:
        rating = 1  # 非常差
    elif prediction < 0.4:
        rating = 2  # 较差
    elif prediction < 0.6:
        rating = 3  # 一般
    elif prediction < 0.8:
        rating = 4  # 较好
    else:
        rating = 5  # 非常好
    
    return category, rating

# 示例评分
# image_path = 'data/validation/日出/20231205152729CA1A4A1999E0AD2D6773E7CC2818A5FA.jpg'
image_path = 'data/validation/云海/300.jpeg'
category, rating = rate_scenery(image_path)
print(f'Image Category: {category}')
print(f'Image Rating: {rating}')
