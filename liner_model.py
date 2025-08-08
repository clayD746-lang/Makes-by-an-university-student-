import numpy as np
import pandas as pd
import keras
import ml_edu.results
import ml_edu.experiment
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

## @Author ClayD in SH.Sumhs

##此代码为线性回归模型，原数据集为chicago_taxi_train

##通过函数定义了线性回归模型和训练模型，按4:1的比例划分训练集和测试集，最后计算R²分数、RMSE、以及平均绝对百分比误差来评估模型好坏



dataframe = pd.read_csv(r"C:\Users\1\Downloads\chicago_taxi_train.csv")#文件链接
print(f'一共有 {len(dataframe)}','行数据')

training_df = dataframe[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
correlations = training_df.corr(numeric_only=True)
print(correlations)

first = 'The most important feature is miles '
print(first)

def liner_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)
  return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)
  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )
print("SUCCESS: defining linear regression functions complete.")

train_df, test_df = train_test_split(training_df, test_size=0.2, random_state=42)
print(f"训练集大小: {len(train_df)} 行")
print(f"测试集大小: {len(test_df)} 行")

settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES', 'TRIP_SECONDS']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_1 = liner_model(settings_1, metrics)

experiment_1 = train_model('two_features', model_1, train_df, 'FARE', settings_1)


ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])


print("\n测试集上的预测结果:")
ml_edu.results.plot_model_predictions(experiment_1, test_df, 'FARE')


loss_history = experiment_1.metrics_history['loss']
fig = px.line(
    x=range(1, len(loss_history) + 1),
    y=loss_history,
    labels={'x': 'Epoch', 'y': 'Loss'},
    title='Training Loss Curve'
)
fig.update_layout(
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    template='plotly_white'
)
fig.show()

print("\n模型评估:")
test_features = {name: test_df[name].values for name in settings_1.input_features}
test_labels = test_df['FARE'].values
predictions = model_1.predict(test_features).flatten()

absolute_percentage_errors = np.abs((test_labels - predictions) / test_labels)
mape = np.mean(absolute_percentage_errors) * 100
accuracy = 100 - mape
print(f"模型准确率: {accuracy:.2f}% (基于平均绝对百分比误差)")

rmse = np.sqrt(np.mean((test_labels - predictions) ** 2))
print(f"均方根误差 (RMSE): {rmse:.2f}")