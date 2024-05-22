# Foraging Swarm Simulator

## Сборка ARGoS-контроллеров

```
mkdir controller/build
cd controller/build
cmake -DCMAKE_CXX_FLAGS="-std=c++17" ..
cd ..
make
```

## Установка зависимостей
```
pip install -r requirements.txt
```

## Запуск обучения агентов
```
python3 ai/tianshou_train.py
```

## Запуск визуализации действий обученных агентов
```
python3 ai/tianshou_test.py
```
