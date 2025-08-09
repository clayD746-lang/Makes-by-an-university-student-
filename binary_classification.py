import numpy as np
import pandas as pd
import plotly.express as px
import io
import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import ml_edu.experiment
import ml_edu.results

## @Author ClayD in Sh.Sumhs at 2025.8.9

# 这个模型是基于一份含有两种类型大米，共七个特征的数据集制作。归一化后划分训练集测试集和验证集。这里提取标签单独存放是因为考虑到多轮训练，带标签进去可能会导致标签泄露。
# 接下来定义两个函数，调整超参，计算性能指标，绘图，输出训练集和测试集的比较结果  完事。



print("Start the programme")

# 读取数据
dataframe = pd.read_csv(r"C:\Users\1\Downloads\Rice_Cammeo_Osmancik.csv")

# 归一化处理
feature_mean = dataframe.mean(numeric_only=True)
feature_std = dataframe.std(numeric_only=True)
numerical_features = dataframe.select_dtypes('number').columns
normalized_dataset = (
    dataframe[numerical_features] - feature_mean
) / feature_std

# 防止标签丢失
normalized_dataset['Class'] = dataframe['Class']

normalized_dataset['Class_Bool'] = np.where(normalized_dataset['Class'] == 'Cammeo', 1, 0)

# 划分训练集，验证集和测试集
rows = len(normalized_dataset)
trains = round(rows * 0.8)
index_90th = trains + round(rows * 0.1)
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:trains]
validation_data = shuffled_dataset.iloc[trains:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

label_columns = ['Class', 'Class_Bool']
train_features = train_data.drop(columns=label_columns)
train_label = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_label = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

all_input_features = [
  'Eccentricity',
  'Major_Axis_Length',
  'Minor_Axis_Length',
  'Area',
  'Convex_Area',
  'Perimeter',
  'Extent',
]

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    model_inputs = [
        keras.Input(name=feature, shape=(1,))
        for feature in settings.input_features
    ]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(
        units=1, name='dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:

    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
        verbose=1,  # 显示训练进度
    )

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )
settings_all_features = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.5,
    input_features=all_input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy',
        threshold=settings_all_features.classification_threshold,
    ),
    keras.metrics.Precision(
        name='precision',
        thresholds=settings_all_features.classification_threshold,
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model_all_features = create_model(settings_all_features, metrics)

# Train the model on the training set.
experiment_all_features = train_model(
    'all features',
    model_all_features,
    train_features,
    train_label,
    settings_all_features,
)

ml_edu.results.plot_experiment_metrics(
    experiment_all_features, ['accuracy', 'precision', 'recall']
)
plt.show()
ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])
plt.show()

def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
  print('Comparing metrics between train and test:')
  for metric, test_value in test_metrics.items():
    print('------')
    print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
    print(f'Test {metric}:  {test_value:.4f}')


test_metrics_all_features = experiment_all_features.evaluate(
    test_features,
    test_labels,
)
compare_train_test(experiment_all_features, test_metrics_all_features)