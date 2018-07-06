import sys
import os
from fruit_model import FruitGanModel


model = FruitGanModel(batch_size=10, epochs=10, image_size=32)
model.prepare_train_data()
model.train()
model.evaluate()

fruit = model.predict("./data/test/apple/t4.jpg")
print('fruit is ', fruit)
