# Распознавание лиц. CV2 #

```
find.py - распознает лицо на потоке с камеры и выделяет его в прямоугольник. Никакого обучения. Просто детект

face_dataset.py - Создание датасета для обучения. 
Требуется ввести id пользователя (только числа!)

face_training.py - Обучение модели trainer/trainer.yml

face_detect.py - Ищет на потоке с камеры лица, и, опираясь на датасет, определяет личности. 
Обводит лицо в прямоугольник и подписывает его
После задания нового пользователя требуется добавить его в массив names = [] (строка 17)
```