from src.inference import ImageClassifier

classifier = ImageClassifier(model_type='resnet18', num_classes=10)
result = classifier.predict('C:/Users/bondi/Pictures/dog.jpg')  # <- change this to a real image on your PC
print(result)