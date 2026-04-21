from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model("models/eye_model.h5")

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "dataset/eyes/test",
    target_size=(224,224),
    class_mode='categorical'
)

loss, acc = model.evaluate(test_data)

print("✅ Test Accuracy:", acc)