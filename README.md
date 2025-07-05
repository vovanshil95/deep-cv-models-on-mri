# **DeepSymNet: Обнаружение ишемического инсульта через анализ симметрии полушарий мозга**

[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Репозиторий реализует модели глубокого обучения для обнаружения ишемического инсульта на FLAIR МРТ-сканах путём анализа симметрии между полушариями мозга. Подход сочетает 2D/3D-CNN и топологический анализ данных для интерпретируемой диагностики.

---

## **Основные особенности**

- **Двойная архитектура** для анализа 3D-объёмов и 2D-срезов
- **Топологические штрих-коды устойчивости** для анализа активаций слоёв
- **Grad-CAM тепловые карты** для интерпретируемости моделей
- **Генерация синтетических данных**
- **Высокая точность**: 97.6% на реальных 3D МРТ-данных

---

## **Установка**

**1. Клонируйте репозиторий:**
```bash
git clone https://github.com/vovanshil95/deep-cv-models-on-mri.git
cd deepsymnet
```

**2. Установите зависимости:**
```bash
pip install -r requirements.txt
pip install noise
pip install git+https://github.com/aimclub/eXplain-NNs
```

## **Структура данных:**

```bash
/normalized_hemispheres/
  ├── normal/
  │   ├── patient1/
  │   │   ├── left.nii
  │   │   └── right.nii
  └── pathology/
      └── patientX/
          ├── left.nii
          └── right.nii
```

## **Результаты**

**Сравнение производительности моделей**

| Модель | Точность | ROC-AUC |
| ------ | -------- | ------- | 
| 3D CNN | 97.6%    | 0.995   |
| 2D CNN | 90.5%    | 0.966   |

Топологические метрики

| Слой | Энтропия Рени (α=2) | Отношение длины штрих-кодов (3D / 2D) |
| ---- | ------------------- | ------------------------------------- |
| 1    | 0.500 vs 0.477      | 1.97 vs 3.27                          |
| 2    | 0.395 vs 0.477      | 1.78 vs 5.48                          |
| 3    | 0.428 vs 0.491      | 1.94 vs 9.18                          |
| 4    | 0.450 vs 0.487      | 2.15 vs 13.26                         |


## **Цитирование**

Если вы используете этот проект, пожалуйста, цитируйте:

```bibtex
@misc{deepsymnet2023,
  title={Ischemic Stroke Detection via Symmetry Analysis and Topological Learning},
  author={[Shilonosov Vladimir]},
  year={2023},
  publisher={GitHub},
  howpublished={\url{https://github.com/vovanshil95/deep-cv-models-on-mri}}
}
```
