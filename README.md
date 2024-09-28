# Handwriting-recognition-AI Model

 
손으로 쓴 숫자와 연산 기호를 인식하고, 이를 기반으로 간단한 연산을 수행하는 딥러닝 모델

이 모델은 CNN(Convolutional Neural Network)와 FCN(Fully Connected Network)를 결합하여 구성됩니다.

 CNN이 이미지 인식을, FCN이 연산 처리를 담당합니다.
 
![roadmap.png](https://github.com/yeonhochi/Handwriting-recognition-AI/blob/main/roadmap.png)

----
## CNN1, CNN2:

+ **Input**: 28x28 images of handwritten digits (0-9)

+ **Output**: A 10-bit one-hot encoded representation of the recognized digit

----
## CNN3:

+ **Input**: 28x28 images of handwritten arithmetic operators (+, -, ×)

+ **Output**: A 3-bit one-hot encoded representation of the recognized operator

----
## FCN1:

+ **Input**: 10-bit one-hot encoded outputs from CNN1 and CNN2

+ **Output**: A 4-bit binary representation of the digit

----
## FCN2:

+ **Input**: Two 4-bit binary outputs from FCN1 and a 3-bit encoded output from CNN3 (total 11 bits)

+ **Output**: The result of the arithmetic operation in an 8-bit signed binary format

----
## FCN3:

+ **Input**: The 8-bit signed binary result

+ **Output**: Two seven-segment LED digit images displaying the final result

---
이 구조를 통해 손으로 쓴 숫자와 연산 기호를 인식하고

이를 기반으로 연산을 수행하여 최종 결과를 시각적으로 출력할 수 있습니다.
